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

# Two source schemas exist across collide2v productions, selected via
# data_processing.candidate_schema (default "legacy", matching this script's
# original hardcoded behavior):
#
# "legacy" (/collide/collide-1m and other raw COLLIDE2V dumps): every L1T_*
#   collection is a set of FLAT, underscore-joined columns (L1T_PFCand_PT,
#   L1T_MuonTight_Eta, ...) straight off the raw producer, main candidates
#   come from L1T_PFCand, and dxy/dxysig need deriving from D0/ErrorD0 here.
#
# "regionized" (aidascoutrepo's Stage-1 collide2v_regionized output, e.g.
#   collide-1m-ptrawv1.0 -- see aida_scout.data.converters.
#   convert_collide2v_regionized): has no L1T_PFCand at all. Every L1T_*
#   collection is written as a NESTED struct column instead (dot-joined when
#   read back, e.g. "L1T_JetAK4.PT" not "L1T_JetAK4_PT"), and the main
#   candidate collection is L1T_PUPPIPart with fields ALREADY renamed/derived
#   by that pipeline's own gather_and_select_puppi_candidates: pt, eta, phi,
#   dxy (=D0), dxysig (=D0/(ErrorD0+EPS), already computed), pdgId (already
#   collapsed to the unsigned {0,11,13,22,211} bucket scheme), charge -- so
#   this schema's pf branch is read directly, not re-derived.
COLLIDE2V_SCHEMAS = {
    "legacy": {
        "candidate_collection": "L1T_PFCand",
        "nested": False,
        "candidate_derived": False,   # dxy/dxysig computed here from D0/ErrorD0
    },
    "regionized": {
        "candidate_collection": "L1T_PUPPIPart",
        "nested": True,               # every L1T_* collection is a nested struct column
        "candidate_derived": True,    # dxy/dxysig/pdgId already derived at Stage 1
    },
}

def _col(collection: str, field: str, nested: bool) -> str:
    """Dot-joined path for ak.from_parquet's columns= projection (awkward's parquet
    reader accepts "Collection.field" strings there to select a nested sub-field)."""
    return f"{collection}.{field}" if nested else f"{collection}_{field}"

def _get(arr: ak.Array, collection: str, field: str, nested: bool) -> ak.Array:
    """Read a loaded array's field, AFTER ak.from_parquet -- unlike _col's dotted string
    (a columns= projection path), a nested record's sub-field here needs a real second
    __getitem__ (arr["L1T_PUPPIPart"]["pt"]), not the dotted string reused as one key."""
    return arr[collection][field] if nested else arr[f"{collection}_{field}"]

def pf_columns_for(candidate_schema: str) -> list:
    if candidate_schema not in COLLIDE2V_SCHEMAS:
        raise ValueError(f"Unknown candidate_schema {candidate_schema!r}; expected one of {list(COLLIDE2V_SCHEMAS)}.")
    schema = COLLIDE2V_SCHEMAS[candidate_schema]
    nested = schema["nested"]
    cand = schema["candidate_collection"]
    if schema["candidate_derived"]:
        candidate_columns = [_col(cand, f, nested) for f in ("pt", "eta", "phi", "dxy", "dxysig", "pdgId")]
    else:
        candidate_columns = [_col(cand, f, nested) for f in ("PT", "Eta", "Phi", "D0", "ErrorD0", "PID")]
    aux_columns = [
        _col(collection, f, nested)
        for collection in ("L1T_MuonTight", "L1T_Electron", "L1T_PhotonTight")
        for f in ("PT", "Eta", "Phi")
    ]
    return candidate_columns + aux_columns

# Default (legacy) column list -- kept as a module-level constant since it's
# also the schema gather_pfcands' default argument documents.
PF_COLUMNS = pf_columns_for("legacy")

def obj_columns_for(candidate_schema: str) -> list:
    nested = COLLIDE2V_SCHEMAS[candidate_schema]["nested"]
    return [
        _col(collection, f, nested)
        for collection in ("L1T_JetAK4", "L1T_MuonTight", "L1T_Electron", "L1T_PhotonTight")
        for f in ("PT", "Eta", "Phi")
    ] + [_col("L1T_MET", f, nested) for f in ("MET", "Phi")]

# Additional columns needed for the top-k typed-object branch (AE input, axis 1 of double DisCo)
OBJ_COLUMNS = obj_columns_for("legacy")

def load_events(path: str, columns: list, max_events: int = -1) -> ak.Array:
    arr = ak.from_parquet(path, columns=columns)
    if max_events is not None and max_events > 0:
        arr = arr[:max_events]
    # PT/Eta/Phi/Mass are stored as float16 in COLLIDE2V parquet files, but awkward has no
    # compiled argsort/etc. kernels for float16 - upcast every field to float32 up front.
    # ak.values_astype recurses through nested record fields too (the "regionized" schema's
    # L1T_PUPPIPart/L1T_JetAK4/etc. columns), so this works unchanged for both schemas.
    arr = ak.zip({field: ak.values_astype(arr[field], np.float32) for field in arr.fields}, depth_limit=1)
    return arr

def gather_pfcands(arr: ak.Array, candidate_schema: str = "legacy") -> ak.Array:
    """Build the main PF-candidate-level feature set (pt, eta, phi, dxy, dxysig, is_pf, pdgId) from raw L1T objects."""

    schema = COLLIDE2V_SCHEMAS[candidate_schema]
    nested = schema["nested"]
    cand = schema["candidate_collection"]

    def cf(field):
        return _get(arr, cand, field, nested)

    if schema["candidate_derived"]:
        # "regionized": dxy/dxysig/pdgId already computed by Stage 1.
        pf = ak.zip({
            "pt": cf("pt"),
            "eta": cf("eta"),
            "phi": cf("phi"),
            "dxy": cf("dxy"),
            "dxysig": cf("dxysig"),
            "is_pf": ak.ones_like(cf("pt")),
            "pdgId": cf("pdgId"),
        })
    else:
        pf = ak.zip({
            "pt": cf("PT"),
            "eta": cf("Eta"),
            "phi": cf("Phi"),
            "dxy": cf("D0"),
            "dxysig": cf("D0") / (cf("ErrorD0") + EPS),
            "is_pf": ak.ones_like(cf("PT")),
            "pdgId": cf("PID"),
        })

    def aux(collection: str, pdg_id: int) -> ak.Array:
        # L1T_MuonTight/Electron/PhotonTight carry no D0/charge, unlike the main candidate collection
        pt = _get(arr, collection, "PT", nested)
        return ak.zip({
            "pt": pt,
            "eta": _get(arr, collection, "Eta", nested),
            "phi": _get(arr, collection, "Phi", nested),
            "dxy": ak.zeros_like(pt),
            "dxysig": ak.zeros_like(pt),
            "is_pf": ak.zeros_like(pt),
            "pdgId": ak.ones_like(pt) * pdg_id,
        })

    muons = aux("L1T_MuonTight", 13)
    electrons = aux("L1T_Electron", 11)
    photons = aux("L1T_PhotonTight", 22)

    return ak.concatenate([pf, muons, electrons, photons], axis=1)

def gather_objects_for_ae(arr: ak.Array, candidate_schema: str = "legacy") -> ak.Array:
    """Build the top-k typed-object set for the frozen-AE branch: 10 jets, 4 muons, 4 electrons, 4 photons, 1 MET."""

    nested = COLLIDE2V_SCHEMAS[candidate_schema]["nested"]

    def topk_pad(pt: ak.Array, eta: ak.Array, phi: ak.Array, type_id: int, k: int) -> ak.Array:
        order = ak.argsort(pt, axis=1, ascending=False)
        pt_s, eta_s, phi_s = pt[order], eta[order], phi[order]
        pt_p = ak.pad_none(pt_s, k, axis=1, clip=True)
        eta_p = ak.pad_none(eta_s, k, axis=1, clip=True)
        phi_p = ak.pad_none(phi_s, k, axis=1, clip=True)
        typ = ak.ones_like(pt_p) * type_id
        return ak.zip({"pt": pt_p, "eta": eta_p, "phi": phi_p, "type": typ})

    def field(collection, name):
        return _get(arr, collection, name, nested)

    # type_id: 0=jet, 1=muon, 2=electron, 3=photon, 4=MET
    jets = topk_pad(field("L1T_JetAK4", "PT"), field("L1T_JetAK4", "Eta"), field("L1T_JetAK4", "Phi"), type_id=0, k=10)
    muons = topk_pad(field("L1T_MuonTight", "PT"), field("L1T_MuonTight", "Eta"), field("L1T_MuonTight", "Phi"), type_id=1, k=4)
    electrons = topk_pad(field("L1T_Electron", "PT"), field("L1T_Electron", "Eta"), field("L1T_Electron", "Phi"), type_id=2, k=4)
    photons = topk_pad(field("L1T_PhotonTight", "PT"), field("L1T_PhotonTight", "Eta"), field("L1T_PhotonTight", "Phi"), type_id=3, k=4)

    # L1T_MET_* is already a length-1 list per event; pad defensively rather than assume it.
    met_pt = ak.pad_none(field("L1T_MET", "MET"), 1, axis=1, clip=True)
    met_phi = ak.pad_none(field("L1T_MET", "Phi"), 1, axis=1, clip=True)
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
    candidate_schema = cfg.get("candidate_schema", "legacy")
    logger.info(f"Soft-kill cell size: {sk_cell_size}")
    logger.info(f"Also save AE object-level features: {also_save_objects}")
    logger.info(f"Candidate schema: {candidate_schema} ({COLLIDE2V_SCHEMAS[candidate_schema]['candidate_collection']})")

    if split and store_by_class:
        raise ValueError("Cannot use both split and store_by_class options at the same time.")

    pf_columns = pf_columns_for(candidate_schema)
    columns = pf_columns + obj_columns_for(candidate_schema) if also_save_objects else pf_columns

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
                gather_pfcands(arr, candidate_schema=candidate_schema),
                label=label,
                n_objects=n_objects,
                sk_cell_size=sk_cell_size,
                sort_by_pt=sort_by_pt,
            )

            if also_save_objects:
                obj_tensor = process_objects_for_ae(gather_objects_for_ae(arr, candidate_schema=candidate_schema))
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
