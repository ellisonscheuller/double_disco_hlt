import argparse
import datetime
import logging
from pathlib import Path
import os
from tqdm import tqdm
import glob
import numpy as np
import awkward as ak

from converterHLT import process_pfcands, process_objects_for_ae, assemble_and_save
from embedding_pca_epoch.utils.cfg_handler import data_config
from embedding_pca_epoch.utils.data_utils import EPS

# Making it possible to log to file as well as to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("converterCollide")

# Columns for the main PF-candidate branch (raw L1T PFCand + Muon/Electron/Photon)
PF_COLUMNS = [
    "L1T_PFCand_PT", "L1T_PFCand_Eta", "L1T_PFCand_Phi", "L1T_PFCand_D0", "L1T_PFCand_ErrorD0", "L1T_PFCand_PID",
    "L1T_MuonTight_PT", "L1T_MuonTight_Eta", "L1T_MuonTight_Phi",
    "L1T_Electron_PT", "L1T_Electron_Eta", "L1T_Electron_Phi",
    "L1T_PhotonTight_PT", "L1T_PhotonTight_Eta", "L1T_PhotonTight_Phi",
]

# Additional columns needed for the top-k typed-object branch (AE input, axis 1 of double DisCo)
OBJ_COLUMNS = [
    "L1T_JetAK4_PT", "L1T_JetAK4_Eta", "L1T_JetAK4_Phi",
    "L1T_MuonTight_PT", "L1T_MuonTight_Eta", "L1T_MuonTight_Phi",
    "L1T_Electron_PT", "L1T_Electron_Eta", "L1T_Electron_Phi",
    "L1T_PhotonTight_PT", "L1T_PhotonTight_Eta", "L1T_PhotonTight_Phi",
    "L1T_MET_MET", "L1T_MET_Phi",
]

def load_events(path: str, columns: list, max_events: int = -1) -> ak.Array:
    arr = ak.from_parquet(path, columns=columns)
    if max_events is not None and max_events > 0:
        arr = arr[:max_events]
    # PT/Eta/Phi/Mass are stored as float16 in COLLIDE2V parquet files, but awkward has no
    # compiled argsort/etc. kernels for float16 - upcast every field to float32 up front.
    arr = ak.zip({field: ak.values_astype(arr[field], np.float32) for field in arr.fields}, depth_limit=1)
    return arr

def gather_pfcands(arr: ak.Array) -> ak.Array:
    """Build the main PF-candidate-level feature set (pt, eta, phi, dxy, dxysig, is_pf, pdgId) from raw L1T objects."""

    pf = ak.zip({
        "pt": arr["L1T_PFCand_PT"],
        "eta": arr["L1T_PFCand_Eta"],
        "phi": arr["L1T_PFCand_Phi"],
        "dxy": arr["L1T_PFCand_D0"],
        "dxysig": arr["L1T_PFCand_D0"] / (arr["L1T_PFCand_ErrorD0"] + EPS),
        "is_pf": ak.ones_like(arr["L1T_PFCand_PT"]),
        "pdgId": arr["L1T_PFCand_PID"],
    })

    def aux(prefix: str, pdg_id: int) -> ak.Array:
        # L1T_MuonTight/Electron/PhotonTight carry no D0/charge, unlike L1T_PFCand
        pt = arr[f"{prefix}_PT"]
        return ak.zip({
            "pt": pt,
            "eta": arr[f"{prefix}_Eta"],
            "phi": arr[f"{prefix}_Phi"],
            "dxy": ak.zeros_like(pt),
            "dxysig": ak.zeros_like(pt),
            "is_pf": ak.zeros_like(pt),
            "pdgId": ak.ones_like(pt) * pdg_id,
        })

    muons = aux("L1T_MuonTight", 13)
    electrons = aux("L1T_Electron", 11)
    photons = aux("L1T_PhotonTight", 22)

    return ak.concatenate([pf, muons, electrons, photons], axis=1)

def gather_objects_for_ae(arr: ak.Array) -> ak.Array:
    """Build the top-k typed-object set for the frozen-AE branch: 10 jets, 4 muons, 4 electrons, 4 photons, 1 MET."""

    def topk_pad(pt: ak.Array, eta: ak.Array, phi: ak.Array, type_id: int, k: int) -> ak.Array:
        order = ak.argsort(pt, axis=1, ascending=False)
        pt_s, eta_s, phi_s = pt[order], eta[order], phi[order]
        pt_p = ak.pad_none(pt_s, k, axis=1, clip=True)
        eta_p = ak.pad_none(eta_s, k, axis=1, clip=True)
        phi_p = ak.pad_none(phi_s, k, axis=1, clip=True)
        typ = ak.ones_like(pt_p) * type_id
        return ak.zip({"pt": pt_p, "eta": eta_p, "phi": phi_p, "type": typ})

    # type_id: 0=jet, 1=muon, 2=electron, 3=photon, 4=MET
    jets = topk_pad(arr["L1T_JetAK4_PT"], arr["L1T_JetAK4_Eta"], arr["L1T_JetAK4_Phi"], type_id=0, k=10)
    muons = topk_pad(arr["L1T_MuonTight_PT"], arr["L1T_MuonTight_Eta"], arr["L1T_MuonTight_Phi"], type_id=1, k=4)
    electrons = topk_pad(arr["L1T_Electron_PT"], arr["L1T_Electron_Eta"], arr["L1T_Electron_Phi"], type_id=2, k=4)
    photons = topk_pad(arr["L1T_PhotonTight_PT"], arr["L1T_PhotonTight_Eta"], arr["L1T_PhotonTight_Phi"], type_id=3, k=4)

    # L1T_MET_* is already a length-1 list per event; pad defensively rather than assume it.
    met_pt = ak.pad_none(arr["L1T_MET_MET"], 1, axis=1, clip=True)
    met_phi = ak.pad_none(arr["L1T_MET_Phi"], 1, axis=1, clip=True)
    met_eta = ak.zeros_like(met_pt)  # MET has no physical pseudorapidity
    met = ak.zip({"pt": met_pt, "eta": met_eta, "phi": met_phi, "type": ak.ones_like(met_pt) * 4})

    return ak.concatenate([jets, muons, electrons, photons, met], axis=1)

def main(cfg: data_config, overwrite: bool = False):
    """Convert one or more COLLIDE2V (L1T) parquet files to fixed-shape PyTorch tensors and save train/test splits."""

    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

    sample_dir = Path(cfg["sample_dir"]).expanduser()
    n_objects = cfg.get("n_objects", 500)
    nevents_per_class = cfg.get("nevents_per_class", -1)
    pfcands = cfg.get("pfcands", True)
    if not pfcands:
        raise ValueError("pfcands=False (object-level main branch) is not supported for COLLIDE2V yet.")
    sk_cell_size = cfg.get("sk_spacing", None)
    sort_by_pt = cfg.get("sort_by_pt", True)
    store_by_class = cfg.get("store_by_class", False)
    split = cfg.get("split", None)
    also_save_objects = cfg.get("also_save_objects", False)
    logger.info(f"Soft-kill cell size: {sk_cell_size}")
    logger.info(f"Also save AE object-level features: {also_save_objects}")

    if split and store_by_class:
        raise ValueError("Cannot use both split and store_by_class options at the same time.")

    columns = PF_COLUMNS + OBJ_COLUMNS if also_save_objects else PF_COLUMNS

    tensors = {}
    tensors_obj = {} if also_save_objects else None
    file_label_tuples = cfg.get_file_label_map()
    for entry in tqdm(file_label_tuples, desc="Processing files"):
        file_name, label = entry
        file_path = sample_dir / file_name
        file_paths = sorted(glob.glob(os.fspath(file_path))) # Expand possible wildcards

        n_events_left = nevents_per_class
        for path in file_paths:
            arr = load_events(path, columns, max_events=n_events_left)

            event_tensor = process_pfcands(
                gather_pfcands(arr),
                label=label,
                n_objects=n_objects,
                sk_cell_size=sk_cell_size,
                sort_by_pt=sort_by_pt,
            )

            if also_save_objects:
                obj_tensor = process_objects_for_ae(gather_objects_for_ae(arr))
                if obj_tensor.shape[0] != event_tensor.shape[0]:
                    raise RuntimeError(f"Event count mismatch for label {label}: main tensor {event_tensor.shape[0]} vs obj tensor {obj_tensor.shape[0]}")
                tensors_obj[label] = tensors_obj.get(label, []) + [obj_tensor]

            tensors[label] = tensors.get(label, []) + [event_tensor]
            n_events_left -= event_tensor.shape[0]
            if n_events_left <= 0:
                break

    assemble_and_save(
        tensors, tensors_obj, cfg,
        overwrite=overwrite,
        also_save_objects=also_save_objects,
        nevents_per_class=nevents_per_class,
        split=split,
        store_by_class=store_by_class,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config .yaml file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    cfg = data_config(args.config)

    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/converterCollide_{cfg.get_ds_name()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting conversion with config: {args.config}")

    main(cfg, overwrite=args.overwrite)
