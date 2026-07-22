import argparse
import datetime
import logging
from pathlib import Path
import os
from tqdm import tqdm
import glob
import torch
import numpy as np
import uproot
import awkward as ak
from typing import Union

from embedding_pca_epoch.utils.cfg_handler import data_config, join_remote
from embedding_pca_epoch.utils.data_utils import softkill, EPS
uproot.source.xrootd.XRootDSource.timeout = 480

# Making it possible to log to file as well as to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("converterHLT")

def process_particles(
        pt: ak.Array, eta: ak.Array, phi: ak.Array, dxy: ak.Array, btag: ak.Array, has_dxy: ak.Array, has_btag: ak.Array, 
        label: int, 
        n_objects: int = 200,
        sort_by_pt: bool = False,
    ) -> torch.Tensor:
    particles = ak.zip({"pt": pt, "eta": eta, "phi": phi, "dxy": dxy, "btag": btag, "has_dxy": has_dxy, "has_btag": has_btag})
    if sort_by_pt:
        logger.info("Sorting particles by pt")
        particles = particles[ak.argsort(particles.pt, axis=1, ascending=False)]
    counts = ak.num(particles.pt, axis=1)
    clipped_events = counts > n_objects
    if ak.any(clipped_events):
        lost = int(ak.sum(counts[clipped_events] - n_objects))
        logger.warning(f"{int(ak.sum(clipped_events))} events clipped; {lost} particles lost. Increase n_objects to avoid this.")
    padded = ak.pad_none(particles, n_objects, axis=1, clip=True)
    array = np.stack(
        [
            ak.to_numpy(padded["pt"]),
            ak.to_numpy(padded["eta"]),
            ak.to_numpy(padded["phi"]),
            ak.to_numpy(padded["dxy"]),
            ak.to_numpy(padded["btag"]),
            ak.to_numpy(padded["has_dxy"]),
            ak.to_numpy(padded["has_btag"])
        ],
        axis=-1,
    )
    tensor = torch.tensor(array, dtype=torch.float32)
    label_tensor = torch.full((tensor.shape[0], n_objects, 1), label, dtype=torch.float32)
    return torch.cat([tensor, label_tensor], dim=-1)

def gather_particles(tree: uproot.TTree, max_events: int = -1) -> tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
    """Return per-event arrays of pt, eta, phi, pid for the TrigObj objects we keep."""

    # ---- Jets ---------------------
    jet_pt = tree["ScoutingPFJetRecluster_pt"].array(entry_stop=max_events)
    jet_eta = tree["ScoutingPFJetRecluster_eta"].array(entry_stop=max_events)
    jet_phi = tree["ScoutingPFJetRecluster_phi"].array(entry_stop=max_events)
    jet_dxy = ak.zeros_like(jet_pt)  # no dxy for jets
    jet_btag = (tree["ScoutingPFJetRecluster_particleNet_prob_bb"].array(entry_stop=max_events) + tree["ScoutingPFJetRecluster_particleNet_prob_b"].array(entry_stop=max_events)) # Probability of being b-jet
    jet_has_dxy  = ak.zeros_like(jet_pt, dtype=np.int8)  # 0
    jet_has_btag = ak.ones_like(jet_pt,  dtype=np.int8)  # 1
    jets = ak.zip({"pt": jet_pt, "eta": jet_eta, "phi": jet_phi, "dxy": jet_dxy, "btag": jet_btag, "has_dxy": jet_has_dxy, "has_btag": jet_has_btag})

    # ---- Muons -------------------------------------------------------------
    mu_pt = tree["ScoutingMuonVtx_pt"].array(entry_stop=max_events)
    mu_eta = tree["ScoutingMuonVtx_eta"].array(entry_stop=max_events)
    mu_phi = tree["ScoutingMuonVtx_phi"].array(entry_stop=max_events)
    mu_dxy = tree["ScoutingMuonVtx_trk_dxy"].array(entry_stop=max_events)
    mu_btag = ak.zeros_like(mu_pt)  # no b-tag for muons
    # Using dtype that accomodates well true/false values when converted to torch
    mu_has_dxy  = ak.ones_like(mu_pt,  dtype=np.int8)  # 1
    mu_has_btag = ak.zeros_like(mu_pt, dtype=np.int8)  # 0
    muons = ak.zip({"pt": mu_pt, "eta": mu_eta, "phi": mu_phi, "dxy": mu_dxy, "btag": mu_btag, "has_dxy": mu_has_dxy, "has_btag": mu_has_btag})

    # ---- Egamma (electrons + photons) ------------------------------
    e_pt = tree["ScoutingElectron_pt"].array(entry_stop=max_events)
    e_eta = tree["ScoutingElectron_eta"].array(entry_stop=max_events)
    e_phi = tree["ScoutingElectron_phi"].array(entry_stop=max_events)
    e_dxy = ak.zeros_like(e_pt)  # no dxy for electrons
    e_btag = ak.zeros_like(e_pt)  # no b-tag for electrons
    e_has_dxy  = ak.zeros_like(e_pt, dtype=np.int8) # 0
    e_has_btag = ak.zeros_like(e_pt, dtype=np.int8) # 0
    electrons = ak.zip({"pt": e_pt, "eta": e_eta, "phi": e_phi, "dxy": e_dxy, "btag": e_btag, "has_dxy": e_has_dxy, "has_btag": e_has_btag})

    photons_pt = tree["ScoutingPhoton_pt"].array(entry_stop=max_events)
    photons_eta = tree["ScoutingPhoton_eta"].array(entry_stop=max_events)
    photons_phi = tree["ScoutingPhoton_phi"].array(entry_stop=max_events)
    photons_dxy = ak.zeros_like(photons_pt)  # no dxy for photons
    photons_btag = ak.zeros_like(photons_pt)  # no b-tag for photons
    photons_has_dxy  = ak.zeros_like(photons_pt, dtype=np.int8) # 0
    photons_has_btag = ak.zeros_like(photons_pt, dtype=np.int8) # 0
    photons = ak.zip({"pt": photons_pt, "eta": photons_eta, "phi": photons_phi, "dxy": photons_dxy, "btag": photons_btag, "has_dxy": photons_has_dxy, "has_btag": photons_has_btag})

    # ---- Energy sums --------------

    ESum_pt = tree["ScoutingMET_pt"].array(entry_stop=max_events)
    ESum_phi = tree["ScoutingMET_phi"].array(entry_stop=max_events)
    ESum_eta = ak.zeros_like(ESum_pt)  # no well-defined eta; keep 0
    ESum_dxy = ak.zeros_like(ESum_pt)  # no dxy for Esum
    Esum_btag = ak.zeros_like(ESum_pt)  # no b-tag for Esum
    ESum_has_dxy  = ak.zeros_like(ESum_pt, dtype=np.int8) # 0
    ESum_has_btag = ak.zeros_like(ESum_pt, dtype=np.int8) # 0
    ESum = ak.zip({"pt": ESum_pt, "eta": ESum_eta, "phi": ESum_phi, "dxy": ESum_dxy, "btag": Esum_btag,"has_dxy": ESum_has_dxy, "has_btag": ESum_has_btag})
    ESum = ESum[:, np.newaxis]

    # ---- Concatenate everything  --------------------
    combined = ak.concatenate([jets, muons, electrons, photons, ESum], axis=1)

    return combined["pt"], combined["eta"], combined["phi"], combined["dxy"], combined["btag"], combined["has_dxy"], combined["has_btag"]    

def construct_pf_features(branches: ak.Array) -> ak.Array:
    """
    Branches:
        - pt
        - eta
        - phi
        - dxy
        - dxysig
        - pdgId (includes sign for charge)
        - is_pf
    """

    # PF candidates
    branches["ScoutingParticle_is_pf"] = ak.ones_like(branches["ScoutingParticle_pt"])

    # ScoutingMuonNoVtx
    branches["ScoutingMuonNoVtx_dxy"] = branches["ScoutingMuonNoVtx_trk_dxy"]
    branches["ScoutingMuonNoVtx_dxysig"] = branches["ScoutingMuonNoVtx_trk_dxy"] / (branches["ScoutingMuonNoVtx_trk_dxyError"] + EPS)
    branches["ScoutingMuonNoVtx_pdgId"] = branches["ScoutingMuonNoVtx_charge"] * 13
    branches["ScoutingMuonNoVtx_is_pf"] = ak.zeros_like(branches["ScoutingMuonNoVtx_pt"])

    # ScoutingElectron
    branches["ScoutingElectron_dxy"] = -branches["ScoutingElectron_bestTrack_d0"]  # d0 = -dxy
    branches["ScoutingElectron_dxysig"] = ak.zeros_like(branches["ScoutingElectron_pt"]) # dxysig = 0 since no error provided
    branches["ScoutingElectron_pdgId"] = branches["ScoutingElectron_bestTrack_charge"] * 11
    branches["ScoutingElectron_is_pf"] = ak.zeros_like(branches["ScoutingElectron_pt"])

    # ScoutingPhoton
    branches["ScoutingPhoton_dxy"] = ak.zeros_like(branches["ScoutingPhoton_pt"])
    branches["ScoutingPhoton_dxysig"] = ak.zeros_like(branches["ScoutingPhoton_pt"])
    branches["ScoutingPhoton_pdgId"] = ak.ones_like(branches["ScoutingPhoton_pt"]) * 22
    branches["ScoutingPhoton_is_pf"] = ak.zeros_like(branches["ScoutingPhoton_pt"])

    return branches

def gather_pfcands(tree: uproot.TTree, max_events: int = -1) -> ak.Array:
    """Return per-event tensors of PF candidates"""

    branches = tree.arrays([
        # PF candidates
        "ScoutingParticle_pt",
        "ScoutingParticle_eta",
        "ScoutingParticle_phi",
        "ScoutingParticle_dxy",
        "ScoutingParticle_dxysig",
        "ScoutingParticle_pdgId",
        # Muons No Vtx
        "ScoutingMuonNoVtx_pt",
        "ScoutingMuonNoVtx_eta",
        "ScoutingMuonNoVtx_phi",
        "ScoutingMuonNoVtx_trk_dxy",
        "ScoutingMuonNoVtx_trk_dxyError",
        "ScoutingMuonNoVtx_charge",
        # Electrons
        "ScoutingElectron_pt",
        "ScoutingElectron_eta",
        "ScoutingElectron_phi",
        "ScoutingElectron_bestTrack_d0", # d0 = -dxy
        "ScoutingElectron_bestTrack_charge",
        # Photons
        "ScoutingPhoton_pt",
        "ScoutingPhoton_eta",
        "ScoutingPhoton_phi",
    ], entry_stop=max_events)

    branches = construct_pf_features(branches)

    branch_prefixes = [
        "ScoutingParticle_",
        "ScoutingMuonNoVtx_",
        "ScoutingElectron_",
        "ScoutingPhoton_",
    ]
    feature_names = [
        "pt", 
        "eta", 
        "phi", 
        "dxy", 
        "dxysig",
        "is_pf",
        "pdgId",
    ]

    combined = ak.concatenate([
        ak.zip({ftr_name: branches[prefix + ftr_name] for ftr_name in feature_names})
        for prefix in branch_prefixes
    ], axis=1)

    # Log name of fields in combiend ak arr
    logger.info("PF candidate fields: " + ", ".join(combined.fields))

    return combined

def process_pfcands(
        combined: ak.Array,
        label: int, 
        n_objects: int = 200, 
        sk_cell_size: Union[float, None] = None,
        sort_by_pt: bool = False,
    ) -> torch.Tensor:

    if sort_by_pt:
        logger.info("Sorting PF candidates by pt")
        combined = combined[ak.argsort(combined.pt, axis=1, ascending=False)]

    counts = ak.num(combined.pt, axis=1)
    clipped_events = counts > n_objects
    if ak.any(clipped_events):
        lost = int(ak.sum(counts[clipped_events] - n_objects))
        logger.warning(f"{int(ak.sum(clipped_events))} events clipped; {lost} particles lost. Increase n_objects to avoid this.")
    padded = ak.pad_none(combined, n_objects, axis=1, clip=True)
    array = np.stack(
        [ak.to_numpy(padded[field]) for field in padded.fields],
        axis=-1,
    )
    tensor = torch.tensor(array, dtype=torch.float32)
    label_tensor = torch.full((tensor.shape[0], n_objects, 1), label, dtype=torch.float32)

    if sk_cell_size is not None:
        tensor = softkill(tensor, cell_size=sk_cell_size)

    return torch.cat([tensor, label_tensor], dim=-1)

def gather_objects_for_ae(tree: uproot.TTree, max_events: int = -1) -> ak.Array:
    """Return per-event top-k typed objects (pt, eta, phi, type_id) for the frozen-AE branch (axis 1 of double DisCo)."""

    def topk_pad(pt: ak.Array, eta: ak.Array, phi: ak.Array, type_id: int, k: int) -> ak.Array:
        order = ak.argsort(pt, axis=1, ascending=False)
        pt_s, eta_s, phi_s = pt[order], eta[order], phi[order]
        pt_p = ak.pad_none(pt_s, k, axis=1, clip=True)
        eta_p = ak.pad_none(eta_s, k, axis=1, clip=True)
        phi_p = ak.pad_none(phi_s, k, axis=1, clip=True)
        typ = ak.ones_like(pt_p) * type_id
        return ak.zip({"pt": pt_p, "eta": eta_p, "phi": phi_p, "type": typ})

    # type_id: 0=jet, 1=muon, 2=electron, 3=photon, 4=MET
    jet_pt = tree["ScoutingPFJetRecluster_pt"].array(entry_stop=max_events)
    jet_eta = tree["ScoutingPFJetRecluster_eta"].array(entry_stop=max_events)
    jet_phi = tree["ScoutingPFJetRecluster_phi"].array(entry_stop=max_events)
    jets = topk_pad(jet_pt, jet_eta, jet_phi, type_id=0, k=10)

    mu_pt = tree["ScoutingMuonVtx_pt"].array(entry_stop=max_events)
    mu_eta = tree["ScoutingMuonVtx_eta"].array(entry_stop=max_events)
    mu_phi = tree["ScoutingMuonVtx_phi"].array(entry_stop=max_events)
    muons = topk_pad(mu_pt, mu_eta, mu_phi, type_id=1, k=4)

    e_pt = tree["ScoutingElectron_pt"].array(entry_stop=max_events)
    e_eta = tree["ScoutingElectron_eta"].array(entry_stop=max_events)
    e_phi = tree["ScoutingElectron_phi"].array(entry_stop=max_events)
    electrons = topk_pad(e_pt, e_eta, e_phi, type_id=2, k=4)

    g_pt = tree["ScoutingPhoton_pt"].array(entry_stop=max_events)
    g_eta = tree["ScoutingPhoton_eta"].array(entry_stop=max_events)
    g_phi = tree["ScoutingPhoton_phi"].array(entry_stop=max_events)
    photons = topk_pad(g_pt, g_eta, g_phi, type_id=3, k=4)

    met_pt = tree["ScoutingMET_pt"].array(entry_stop=max_events)
    met_phi = tree["ScoutingMET_phi"].array(entry_stop=max_events)
    met_eta = ak.zeros_like(met_pt)
    met_pt, met_eta, met_phi = met_pt[:, np.newaxis], met_eta[:, np.newaxis], met_phi[:, np.newaxis]
    met = ak.zip({"pt": met_pt, "eta": met_eta, "phi": met_phi, "type": ak.ones_like(met_pt) * 4})

    return ak.concatenate([jets, muons, electrons, photons, met], axis=1)

def process_objects_for_ae(combined: ak.Array) -> torch.Tensor:
    """Returns an (N, 23, 4) float32 tensor: (pt, eta, phi, type_id) — the AE input for double DisCo's axis 1."""
    array = np.stack(
        [
            ak.to_numpy(combined["pt"]),
            ak.to_numpy(combined["eta"]),
            ak.to_numpy(combined["phi"]),
            ak.to_numpy(combined["type"]),
        ],
        axis=-1,
    )
    return torch.tensor(array, dtype=torch.float32)

def save_dataset(out_file: Path, full_tensor: torch.Tensor, obj_tensor: Union[torch.Tensor, None] = None) -> None:
    if obj_tensor is None:
        torch.save(full_tensor, os.fspath(out_file))
    else:
        pf = full_tensor[..., :-1]
        label = full_tensor[:, 0, -1].long()
        torch.save({"pf": pf, "label": label, "obj": obj_tensor}, os.fspath(out_file))

def main(cfg: data_config, overwrite: bool = False):
    """Convert one or more ROOT files to fixed-shape PyTorch tensors and save train/test splits."""

    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

    sample_dir = Path(cfg["sample_dir"]).expanduser()
    redir = cfg.get("redir", "")
    n_objects = cfg.get("n_objects", 500)
    nevents_per_class = cfg.get("nevents_per_class", -1)
    pfcands = cfg.get("pfcands", True)
    sk_cell_size = cfg.get("sk_spacing", None)
    sort_by_pt = cfg.get("sort_by_pt", True)
    store_by_class = cfg.get("store_by_class", False)
    split = cfg.get("split", None)
    also_save_objects = cfg.get("also_save_objects", False)
    logger.info(f"PFCands mode: {pfcands}")
    logger.info(f"Soft-kill cell size: {sk_cell_size}")
    logger.info(f"Also save AE object-level features: {also_save_objects}")

    if split and store_by_class:
        raise ValueError("Cannot use both split and store_by_class options at the same time.")

    tensors = {}
    tensors_obj = {} if also_save_objects else None
    file_label_tuples = cfg.get_file_label_map()
    for entry in tqdm(file_label_tuples, desc="Processing files"):
        file_name, label = entry
        file_path = sample_dir / file_name
        # Dont use glob if redir is set
        file_paths = [Path(p) for p in glob.glob(os.fspath(file_path))] if not redir else [file_path]

        n_events_left = nevents_per_class
        for path in file_paths: # Expand possible wildcards
            src = os.fspath(path) if not redir else join_remote(redir, path)
            tree = uproot.open(src)["Events"]

            if pfcands:
                event_tensor = process_pfcands(
                    gather_pfcands(tree, max_events=n_events_left),
                    label=label,
                    n_objects=n_objects,
                    sk_cell_size=cfg.get("sk_spacing", None),
                    sort_by_pt=sort_by_pt,
                )

            else:
                pt, eta, phi, dxy, btag, has_dxy, has_btag = gather_particles(tree, max_events=n_events_left)
                event_tensor = process_particles(
                    pt, eta, phi, dxy, btag, has_dxy, has_btag,
                    label=label,
                    n_objects=n_objects,
                )

            if also_save_objects:
                obj_tensor = process_objects_for_ae(gather_objects_for_ae(tree, max_events=n_events_left))
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

def assemble_and_save(
        tensors: dict,
        tensors_obj: Union[dict, None],
        cfg: data_config,
        overwrite: bool,
        also_save_objects: bool,
        nevents_per_class: int,
        split,
        store_by_class: bool,
    ) -> None:
    """Concatenate per-class chunks, shuffle (pf/label/obj kept aligned), and write output file(s)."""

    # .pop() instead of .items() so each label's raw per-file chunks are freed the instant
    # they're concatenated, instead of staying alive (referenced by main()'s `tensors` dict)
    # for the rest of this function.
    class_tensors = {}
    for label in list(tensors.keys()):
        class_tensors[label] = torch.cat(tensors.pop(label), dim=0)
    class_tensors_obj = {}
    if also_save_objects:
        for label in list(tensors_obj.keys()):
            class_tensors_obj[label] = torch.cat(tensors_obj.pop(label), dim=0)

    for label in class_tensors:
        if nevents_per_class > 0:
            class_tensors[label] = class_tensors[label][:nevents_per_class]
            if also_save_objects:
                class_tensors_obj[label] = class_tensors_obj[label][:nevents_per_class]

    total_num_events = sum(tensor.shape[0] for tensor in class_tensors.values())
    logger.info("Class event counts:")
    for label in class_tensors:
        logger.info(f"  Label {label}: {class_tensors[label].shape[0]} events ({round(class_tensors[label].shape[0] / total_num_events * 100, 2)}%)")

    full_tensor = torch.cat([class_tensors[label] for label in class_tensors], dim=0)
    class_tensors.clear()
    full_tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    full_obj_tensor = None
    if also_save_objects:
        full_obj_tensor = torch.cat([class_tensors_obj[label] for label in class_tensors_obj], dim=0)
        class_tensors_obj.clear()
        full_obj_tensor.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        logger.info(f"AE object feature tensor shape: {tuple(full_obj_tensor.shape)}")

    perm = torch.randperm(full_tensor.shape[0]) # Randomize order
    full_tensor = full_tensor[perm]
    if also_save_objects:
        full_obj_tensor = full_obj_tensor[perm]

    output_prefix = cfg.get_ds_name()
    if output_prefix == "":
        output_prefix = "embedding_hlt_ssl"
        logger.warning(f"Dataset name not found in config; using default {output_prefix}.")

    out_path = Path(cfg.get("out_path", "./")).expanduser()
    os.makedirs(out_path, exist_ok=True)

    if split is not None:
        # One file per train/test split, each w/ mixed classes
        split_idx = int(cfg.get("split", 0.8) * full_tensor.shape[0])
        train_fname = out_path / f"{output_prefix}_train.pt"
        test_fname = out_path / f"{output_prefix}_test.pt"
        file_exists = train_fname.exists() or test_fname.exists()
        if file_exists and not overwrite:
            raise FileExistsError(f"Output files with prefix {output_prefix} already exist in {os.fspath(out_path)}. Use --overwrite to overwrite.")
        train_tensor = full_tensor[:split_idx]
        test_tensor = full_tensor[split_idx:]

        n_events_per_class_train = torch.bincount(train_tensor[:, 0, -1].to(torch.int64))
        n_events_per_class_test = torch.bincount(test_tensor[:, 0, -1].to(torch.int64))
        logger.info("Train class event counts:")
        for i in range(len(n_events_per_class_train)):
            logger.info(f"  Label {i}: {n_events_per_class_train[i]} events ({round(n_events_per_class_train[i].item() / train_tensor.shape[0] * 100, 2)}%)")
        logger.info("Test class event counts:")
        for i in range(len(n_events_per_class_test)):
            logger.info(f"  Label {i}: {n_events_per_class_test[i]} events ({round(n_events_per_class_test[i].item() / test_tensor.shape[0] * 100, 2)}%)")

        if also_save_objects:
            save_dataset(train_fname, train_tensor, full_obj_tensor[:split_idx])
            save_dataset(test_fname, test_tensor, full_obj_tensor[split_idx:])
        else:
            torch.save(train_tensor, os.fspath(train_fname))
            torch.save(test_tensor, os.fspath(test_fname))
    elif store_by_class:
        # One file per class
        label_name_map = cfg.get_label_name_map()
        event_labels = full_tensor[:, 0, -1].to(torch.int64)
        for label, name in label_name_map.items():
            full_fname = out_path / f"{output_prefix}_{name}_testds.pt"
            if not overwrite and full_fname.exists():
                raise FileExistsError(f"Output file with prefix {output_prefix}_{name}_testds.pt already exists in {os.fspath(out_path)}. Use --overwrite to overwrite.")
            mask = event_labels == int(label)
            class_tensor = full_tensor[mask]
            if also_save_objects:
                save_dataset(full_fname, class_tensor, full_obj_tensor[mask])
            else:
                torch.save(class_tensor, os.fspath(full_fname))
    else:
        # All classes mixed in one file
        full_fname = out_path / f"{output_prefix}.pt"
        if not overwrite and full_fname.exists():
            raise FileExistsError(f"Output file with prefix {output_prefix}.pt already exists in {os.fspath(out_path)}. Use --overwrite to overwrite.")
        if also_save_objects:
            save_dataset(full_fname, full_tensor, full_obj_tensor)
        else:
            torch.save(full_tensor, os.fspath(full_fname))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config .yaml file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    cfg = data_config(args.config)

    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/converterHLT_{cfg.get_ds_name()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting conversion with config: {args.config}")

    main(cfg, overwrite=args.overwrite)
