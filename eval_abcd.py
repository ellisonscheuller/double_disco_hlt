import os
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic, gaussian_kde
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

from embedding_pca_epoch.models import TransformerEncoder, Projector
from embedding_pca_epoch.autoencoder import Autoencoder
from embedding_pca_epoch.preprocs import PFPreProcessor
from embedding_pca_epoch.utils.data_utils import load_data

_SIG_COLORS = ["tab:purple", "tab:brown", "tab:olive", "tab:pink", "tab:cyan"]


def _load_pt(data_or_path):
    if isinstance(data_or_path, str):
        return torch.load(data_or_path, map_location="cpu")
    return data_or_path


def load_eval_data(test_pt_path, qcd_pt_path=None):
    """
    Build the background eval dataset.

    Default (qcd_pt_path=None): returns test_pt_path unchanged.

    With qcd_pt_path: strips QCD (label==1) from test_pt_path and replaces it
    with events from qcd_pt_path (labels overwritten to 1).  No training overlap
    because test_pt_path should be smcocktail_test.pt (separate file from
    smcocktail_train.pt), and qcd_pt_path should be the held-out mquinna test
    slice that was never seen during training.
    """
    if qcd_pt_path is None:
        return test_pt_path

    bkg = torch.load(test_pt_path, map_location="cpu")
    non_qcd = bkg["label"] != 1
    print(f"  Non-QCD events from {test_pt_path}: {int(non_qcd.sum())}", flush=True)

    qcd = torch.load(qcd_pt_path, map_location="cpu")
    n_qcd = qcd["pf"].shape[0]
    print(f"  QCD events from {qcd_pt_path}: {n_qcd} (relabeled to 1)", flush=True)

    merged = {
        "pf":    torch.cat([bkg["pf"][non_qcd],  qcd["pf"]],  dim=0),
        "obj":   torch.cat([bkg["obj"][non_qcd],  qcd["obj"]], dim=0),
        "label": torch.cat([bkg["label"][non_qcd],
                             torch.ones(n_qcd, dtype=bkg["label"].dtype)], dim=0),
    }
    print(f"  Total eval events: {merged['pf'].shape[0]}", flush=True)
    return merged

# Map filename keywords → physics display names
_SIGNAL_DISPLAY_NAMES = {
    "TpTp":     "T'T'",
    "VBFHto2B": "VBF H$\\to$b$\\bar{b}$",
    "VBF":      "VBF",
    "ttbar":    r"$t\bar{t}$",
    "TTbar":    r"$t\bar{t}$",
    "tttt":     r"$t\bar{t}t\bar{t}$",
}

def _signal_display_name(label: str, pt_path: str = "") -> str:
    """Return a physics display name for a signal, using filename if label is generic."""
    for key, display in _SIGNAL_DISPLAY_NAMES.items():
        if key in label or key in pt_path:
            return display
    return label

# ─────────────────────────────────────────────
# ABCD helpers
# ─────────────────────────────────────────────

def abcd_counts(loss_1, loss_2, percent_1, percent_2):
    thresh_1 = np.quantile(loss_1, percent_1)
    thresh_2 = np.quantile(loss_2, percent_2)
    A = int(((loss_1 > thresh_1) & (loss_2 > thresh_2)).sum())
    B = int(((loss_1 > thresh_1) & (loss_2 <= thresh_2)).sum())
    C = int(((loss_1 <= thresh_1) & (loss_2 > thresh_2)).sum())
    D = int(((loss_1 <= thresh_1) & (loss_2 <= thresh_2)).sum())
    return thresh_1, thresh_2, A, B, C, D


def nonclosure_A(A, B, C, D, eps=1e-8):
    A_hat = (B * C) / max(D, eps)
    if A_hat <= 0:
        return np.inf, A_hat
    return (A - A_hat) / A_hat, A_hat


def profile_plot(ax, x, y, nbins=30, logx=False, min_per_bin=20, label="mean ± SE"):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if logx:
        m &= (x > 0)
    x, y = x[m], y[m]

    xu = np.log10(x) if logx else x
    lo, hi = float(xu.min()), float(xu.max())
    if lo == hi:
        hi = np.nextafter(hi, np.inf)
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean, _, _ = binned_statistic(xu, y, statistic="mean",  bins=edges)
    std,  _, _ = binned_statistic(xu, y, statistic="std",   bins=edges)
    cnt,  _, _ = binned_statistic(xu, y, statistic="count", bins=edges)
    sem = std / np.sqrt(np.maximum(cnt, 1))

    good = cnt >= min_per_bin
    xc = centers[good]
    xplot = (10.0 ** xc) if logx else xc
    if logx:
        ax.set_xscale("log")

    ax.errorbar(xplot, mean[good], yerr=sem[good],
                fmt="o", ms=3, lw=1, capsize=2, label=label)
    ax.grid(alpha=0.3)
    return {"x": xplot, "mean": mean[good], "sem": sem[good], "count": cnt[good]}


# ─────────────────────────────────────────────
# AE inference from joint checkpoint
# ─────────────────────────────────────────────

def compute_ae_scores(ckpt, data_or_path, device="cpu", batch_size=4096):
    """
    Loads the AE stored in a joint checkpoint, runs inference on the obj features
    from data_or_path (path string or pre-loaded dict), and returns per-event MSE reco loss.
    """
    ae_sd = ckpt["ae"]

    # infer architecture from state dict
    backbone_linear_keys = sorted(
        k for k in ae_sd if k.startswith("encoder.backbone.") and k.endswith(".weight")
    )
    features  = int(ae_sd[backbone_linear_keys[0]].shape[1])
    enc_nodes = [int(ae_sd[k].shape[0]) for k in backbone_linear_keys]
    latent_dim = int(ae_sd["encoder.fc_latent.weight"].shape[0])
    dec_linear_keys = sorted(
        k for k in ae_sd if k.startswith("decoder.net.") and k.endswith(".weight") and ae_sd[k].ndim == 2
    )
    dec_nodes = [int(ae_sd[k].shape[0]) for k in dec_linear_keys]

    ae = Autoencoder({
        "features":       features,
        "latent_dim":     latent_dim,
        "encoder_config": {"nodes": enc_nodes},
        "decoder_config": {"nodes": dec_nodes},
        "alpha": 1.0,
    }).to(device).eval()
    ae.load_state_dict(ae_sd)
    print(f"  AE: input={features}, enc={enc_nodes}, latent={latent_dim}, dec={dec_nodes}", flush=True)

    # load and normalise obj features the same way as during training
    scaler = ckpt["ae_scaler"]
    mu  = scaler["mu"].cpu().numpy()
    std = scaler["std"].cpu().numpy()

    data = _load_pt(data_or_path)
    obj = data["obj"][:, :, :4].reshape(data["obj"].shape[0], -1).float().numpy()
    obj_norm = torch.from_numpy(((obj - mu) / (std + 1e-8)).astype(np.float32))
    N = obj_norm.shape[0]

    print(f"  Running AE inference on {N} events...", flush=True)
    scores = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = obj_norm[i0:i0 + batch_size].to(device)
            recon, _ = ae(xb)
            mse = ((recon - xb) ** 2).mean(dim=1)
            scores.append(mse.cpu())
    return torch.cat(scores).numpy().astype(np.float32)


# ─────────────────────────────────────────────
# Contrastive model: embed + Mahalanobis
# ─────────────────────────────────────────────

def build_contrastive_model(ckpt, device="cpu"):
    """Build encoder + projector from a loaded checkpoint dict."""
    enc_sd  = ckpt["encoder"]
    proj_sd = ckpt["projector"]

    embed_size = int(enc_sd["cls_token"].shape[-1]) if "cls_token" in enc_sd \
        else int(enc_sd["input_proj.weight"].shape[0])
    latent_dim = int(enc_sd["bottleneck.weight"].shape[0])

    layer_ids = [int(k.split(".")[1]) for k in enc_sd if k.startswith("layers.")]
    num_layers = (max(layer_ids) + 1) if layer_ids else 0

    num_heads = next(
        (int(v.shape[0]) for k, v in enc_sd.items()
         if k.endswith("self_attn.bias_mlp.2.bias")),
        4
    )
    pairwise = ckpt.get("pairwise", False)

    linformer = any("self_attn.e_proj" in k for k in enc_sd)
    if linformer:
        e_proj_key = next(k for k in enc_sd if k.endswith("self_attn.e_proj.weight"))
        w = enc_sd[e_proj_key]
        linear_dim  = int(w.shape[0])
        num_tokens  = int(w.shape[1]) - 1
    else:
        linear_dim = None
        num_tokens = None

    linear_w = sorted([(k, v) for k, v in proj_sd.items()
                        if hasattr(v, "ndim") and v.ndim == 2],
                       key=lambda kv: kv[0])
    proj_dim = int(linear_w[-1][1].shape[0])

    ff_key = next((k for k in enc_sd if k.endswith("linear1.weight")), None)
    dim_feedforward = int(enc_sd[ff_key].shape[0]) if ff_key is not None else 2048

    norm_constants = ckpt.get("norm_constants", {})
    preproc = PFPreProcessor(norm_constants).to(device)

    encoder = TransformerEncoder(
        num_features=preproc.num_features,
        embed_size=embed_size,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        linear_dim=linear_dim,
        num_tokens=num_tokens,
        pairwise=pairwise,
        pre_processor=preproc,
        dim_feedforward=dim_feedforward,
    ).to(device).eval()

    projector = Projector(latent_dim, proj_dim, hidden_dim=(proj_dim * 4)).to(device).eval()
    encoder.load_state_dict(enc_sd)
    projector.load_state_dict(proj_sd)

    print(f"  embed_size={embed_size}, latent_dim={latent_dim}, "
          f"num_heads={num_heads}, num_layers={num_layers}, pairwise={pairwise}", flush=True)
    return encoder, projector


def embed_dataset(encoder, projector, data_or_path, device="cpu", batch_size=512):
    """Run encoder on a .pt file or pre-loaded dict, return (latents [N, D], labels [N])."""
    data   = _load_pt(data_or_path)
    pf     = data["pf"]
    labels = data["label"]
    N = pf.shape[0]
    src = data_or_path if isinstance(data_or_path, str) else "merged dataset"
    print(f"  Running inference on {N} events from {src}...", flush=True)
    latents = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = pf[i0:i0 + batch_size].to(device)
            mask = (xb.abs().sum(-1) == 0)
            mask = torch.cat([
                torch.zeros(mask.size(0), 1, device=device, dtype=torch.bool),
                mask
            ], dim=1)
            latent = encoder(xb, None, mask)
            latents.append(latent.cpu())
    return torch.cat(latents, dim=0).numpy(), labels.numpy()


def _fit_class_transform(embeddings, mask, n_pca, class_name):
    """Fit PCA whitening on embeddings[mask]. Returns (mu, W)."""
    ref = embeddings[mask]
    print(f"  Fitting PCA whitening on {mask.sum()} {class_name} events (dim={ref.shape[1]})...", flush=True)
    mu = ref.mean(axis=0)
    centered = ref - mu
    cov = (centered.T @ centered) / ref.shape[0]
    L, V = np.linalg.eigh(cov)
    if n_pca is not None:
        V = V[:, -n_pca:]
        L = L[-n_pca:]
        print(f"    Using top {n_pca} PCA components", flush=True)
    L = np.clip(L, 1e-6, None)
    W = V / np.sqrt(L)
    return mu, W


def compute_md_scores(ckpt_path, data_or_path, device="cpu", batch_size=512, n_pca=None,
                      bkg_labels=None):
    """
    Runs encoder on data_or_path, fits PCA whitening per background class, and returns
    per-event MD scores.

    bkg_labels: list of class labels to use as background reference.
      - [1]       (default) → existing behaviour: MD from QCD only.
      - [0, 1, 3] → min-MD across DY, QCD, WJets (equal-weight minimum).

    Returns md, labels, mu_qcd, W_qcd, encoder, projector, embeddings, class_transforms.
    class_transforms: list of (label, mu, W) for each bkg class — used for signal inference.
    mu_qcd / W_qcd are the QCD transforms (kept for visualisation regardless of mode).
    """
    if bkg_labels is None:
        bkg_labels = [1]

    _CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}

    print("Loading checkpoint...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder, projector = build_contrastive_model(ckpt, device)

    embeddings, labels = embed_dataset(encoder, projector, data_or_path, device, batch_size)

    # fit per-class transforms
    class_transforms = []
    for cls in bkg_labels:
        mask = (labels == cls)
        if mask.sum() < 10:
            print(f"  WARNING: class {cls} has only {mask.sum()} events — skipping", flush=True)
            continue
        mu, W = _fit_class_transform(embeddings, mask, n_pca, _CLASS_NAMES.get(cls, str(cls)))
        class_transforms.append((cls, mu, W))

    if not class_transforms:
        raise RuntimeError("No background classes with enough events to fit PCA whitening.")

    # compute per-class MD and take element-wise minimum
    md_per_class = []
    for cls, mu_c, W_c in class_transforms:
        z_c = (embeddings - mu_c) @ W_c
        md_per_class.append((z_c * z_c).sum(axis=1))
    md = np.stack(md_per_class, axis=0).min(axis=0).astype(np.float32)

    if len(class_transforms) > 1:
        print(f"  Min-MD across classes {[c for c,_,_ in class_transforms]}", flush=True)

    # QCD transform used for visualisation (corner plots / profile plots)
    qcd_entry = next((t for t in class_transforms if t[0] == 1), class_transforms[0])
    mu_qcd, W_qcd = qcd_entry[1], qcd_entry[2]

    return md, labels, mu_qcd, W_qcd, encoder, projector, embeddings, class_transforms


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def ABCD(config):
    print("Logging in to wandb...", flush=True)
    wandb.login()
    wandb.init(project="AE vs. Contrastive ABCD",
               name=config.get("wandb_run_name", None),
               settings=wandb.Settings(_disable_stats=True),
               config=config)
    run_name = wandb.run.name
    print(f"Run name: {run_name}", flush=True)

    outdir   = config.get("outdir", "outputs_abcd")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── parse signal list ──
    signal_pts = config.get("signal_pt") or []
    signal_labels_cfg = config.get("signal_label") or []
    if isinstance(signal_pts, str):
        signal_pts = [signal_pts]
    if isinstance(signal_labels_cfg, str):
        signal_labels_cfg = [signal_labels_cfg]
    while len(signal_labels_cfg) < len(signal_pts):
        signal_labels_cfg.append(f"Signal{len(signal_labels_cfg) + 1}")

    # ── min-MD mode: use QCD + DY + WJets as background reference classes ──
    bkg_labels = [0, 1, 3] if config.get("min_md") else [1]
    if config.get("min_md"):
        print("Min-MD mode: axis 2 = min(MD_DY, MD_QCD, MD_WJets)", flush=True)

    # ── build eval dataset (optionally replace QCD with mquinna test data) ──
    print("Building eval dataset...", flush=True)
    test_data = load_eval_data(config["contrast_test_pt"], config.get("qcd_pt"))

    # ── load checkpoint once (shared by both AE and contrastive inference) ──
    ckpt = torch.load(config["contrast_ckpt"], map_location=device)

    # ── AE scores: compute from joint checkpoint if available, else load pre-computed ──
    if "ae" in ckpt and "ae_scaler" in ckpt:
        print("Computing AE scores from joint checkpoint...", flush=True)
        ae_bkg = compute_ae_scores(ckpt, test_data, device=device)
    elif config.get("ae_scores_bkg_test"):
        print("Loading pre-computed AE scores...", flush=True)
        ae_bkg = torch.load(config["ae_scores_bkg_test"], map_location="cpu").numpy().astype(np.float32).reshape(-1)
    else:
        raise ValueError("No AE scores available: checkpoint has no 'ae' key and --ae_scores_bkg_test not provided.")

    # free GPU memory from AE before loading contrastive model
    del ckpt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    # reload checkpoint for contrastive model (ckpt was deleted above)
    ckpt = torch.load(config["contrast_ckpt"], map_location=device)

    # ── compute contrastive MD scores ──
    con_bkg, labels, md_mu, md_W, encoder, projector, embeddings_all, class_transforms = compute_md_scores(
        config["contrast_ckpt"],
        test_data,
        device=device,
        n_pca=config.get("n_pca"),
        bkg_labels=bkg_labels,
    )

    if len(con_bkg) != len(ae_bkg):
        raise ValueError(f"Length mismatch: contrastive {len(con_bkg)} vs AE {len(ae_bkg)}")

    # ── mask (finite, positive AE loss) ──
    mask = np.isfinite(ae_bkg) & np.isfinite(con_bkg) & (ae_bkg > 0)
    axis1_bkg = ae_bkg[mask]    # AE reco loss  → x axis (all SM bkg, for plots)
    axis2_bkg = con_bkg[mask]   # contrastive MD → y axis (all SM bkg, for plots)
    labels_masked = labels[mask]
    embeddings_masked = embeddings_all[mask]
    print(f"Events after masking: {mask.sum()}", flush=True)

    # ── Axis 2 score: PCA-whitened MD (fitted on QCD in compute_md_scores) ──
    # con_bkg already contains the PCA-whitened MD for all events.
    # Compute whitened latents for the masked subset (used in corner plots).
    emb_pca = (embeddings_masked - md_mu) @ md_W   # [N, D] whitened latents
    n_pca = emb_pca.shape[1]                        # full latent dim (e.g. 6)
    axis2_pca = axis2_bkg                           # con_bkg is already PCA-whitened MD
    print(f"  PCA whitening applied (dim={n_pca}, fitted on all test QCD)", flush=True)

    # ── QCD-only arrays for ABCD scan and closure ──
    # DisCo was trained to decorrelate only on QCD, so closure is most meaningful on QCD.
    qcd_only  = labels_masked == 1
    axis1_qcd = axis1_bkg[qcd_only]
    axis2_qcd = axis2_pca[qcd_only]  # ABCD/closure on PCA-MD
    print(f"QCD events for ABCD: {qcd_only.sum()}", flush=True)

    # ── signal inference ──
    signals = []
    for sig_pt, sig_label, sig_color in zip(signal_pts, signal_labels_cfg, _SIG_COLORS):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        display_label = _signal_display_name(sig_label, sig_pt)
        print(f"Running {display_label} signal inference...", flush=True)
        sig_embeddings, _ = embed_dataset(encoder, projector, sig_pt, device)
        # compute min-MD across the same background classes used for bkg scoring
        _sig_md_per_class = []
        for _, mu_c, W_c in class_transforms:
            _z_c = (sig_embeddings - mu_c) @ W_c
            _sig_md_per_class.append((_z_c * _z_c).sum(axis=1))
        sig_con = np.stack(_sig_md_per_class, axis=0).min(axis=0).astype(np.float32)
        sig_ae  = compute_ae_scores(ckpt, sig_pt, device=device)
        sig_mask = np.isfinite(sig_ae) & np.isfinite(sig_con) & (sig_ae > 0)
        sig_axis1      = sig_ae[sig_mask]
        sig_axis2      = sig_con[sig_mask]
        sig_emb_masked = sig_embeddings[sig_mask]
        sig_emb_pca    = (sig_emb_masked - md_mu) @ md_W   # QCD whitening for visualisation
        _sig_pca_per_class = []
        for _, mu_c, W_c in class_transforms:
            _z_c = (sig_emb_masked - mu_c) @ W_c
            _sig_pca_per_class.append((_z_c * _z_c).sum(axis=1))
        sig_axis2_pca = np.stack(_sig_pca_per_class, axis=0).min(axis=0).astype(np.float32)
        print(f"  {display_label} events after masking: {sig_mask.sum()}", flush=True)
        signals.append({
            "label":    display_label,
            "color":    sig_color,
            "axis1":    sig_axis1,
            "axis2":    sig_axis2,
            "axis2_pca": sig_axis2_pca,
            "embeddings": sig_emb_masked,
            "emb_pca":  sig_emb_pca,
        })

    # ── ABCD scan ──
    percent = np.linspace(0.75, 0.9995, 60)
    best    = {"nonclosure": np.inf}
    min_A   = int(config.get("min_A", 50))
    min_D   = int(config.get("min_D", 500))

    # nc_grid[i, j] = nonclosure at (p1=percent[i], p2=percent[j]); NaN where the
    # min_A/min_D stats requirement failed, so the 2D scan plot below can show both
    # the closure landscape and where it was masked out.
    nc_grid = np.full((len(percent), len(percent)), np.nan)

    for i, p1 in enumerate(percent):
        for j, p2 in enumerate(percent):
            t1, t2, A, B, C, D = abcd_counts(axis1_qcd, axis2_qcd, p1, p2)
            if A < min_A or D < min_D:
                continue
            nc, A_hat = nonclosure_A(A, B, C, D)
            nc_grid[i, j] = nc
            if np.isfinite(nc) and abs(nc) < abs(best["nonclosure"]):
                best.update(dict(p1=p1, p2=p2, t1=t1, t2=t2,
                                 A=A, B=B, C=C, D=D, A_hat=A_hat, nonclosure=nc))

    if "t1" not in best:
        raise RuntimeError("No ABCD working point found. Try lowering min_A/min_D.")

    t1_opt, t2_opt = best["t1"], best["t2"]
    print(f"Optimized: p1={best['p1']:.3f}, p2={best['p2']:.3f}", flush=True)
    print(f"Thresholds: t1={t1_opt:.4g}, t2={t2_opt:.4g}", flush=True)
    print(f"Nonclosure: {100.0*best['nonclosure']:.2f}%", flush=True)

    wandb.log({
        "ABCD/opt_p1":      best["p1"],
        "ABCD/opt_p2":      best["p2"],
        "ABCD/opt_t1":      float(t1_opt),
        "ABCD/opt_t2":      float(t2_opt),
        "ABCD/nonclosure":  float(best["nonclosure"]),
        "ABCD/A": int(best["A"]), "ABCD/B": int(best["B"]),
        "ABCD/C": int(best["C"]), "ABCD/D": int(best["D"]),
    })

    # ─────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────
    fs, fs_leg, fs_legend = 28, 24, 16
    fig_size   = (8, 6)

    # 2D closure scan - full (p1, p2) grid, colored by |non-closure|, with the
    # optimized working point marked. Blank cells failed the min_A/min_D stats cut.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pct_grid = np.clip(np.abs(nc_grid) * 100.0, 0.0, 100.0)
    mesh = ax.pcolormesh(percent, percent, pct_grid.T, cmap="viridis_r",
                          vmin=0.0, vmax=np.nanpercentile(pct_grid, 95), shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("|Non-closure| (%)", fontsize=fs_leg)
    ax.scatter([best["p1"]], [best["p2"]], marker="*", s=400, color="red",
               edgecolor="black", linewidth=1.0, zorder=5,
               label=f"Optimized: p1={best['p1']:.3f}, p2={best['p2']:.3f}\n"
                     f"|non-closure|={100.0*abs(best['nonclosure']):.2f}%")
    ax.set_xlabel("Percentile threshold, axis 1 (AE reco loss)", fontsize=fs_leg)
    ax.set_ylabel("Percentile threshold, axis 2 (contrastive MD)", fontsize=fs_leg)
    ax.set_title("ABCD closure scan (full grid)", fontsize=fs_leg)
    ax.legend(loc="lower left", fontsize=12, framealpha=0.9)
    fig.tight_layout()
    out_scan2d = os.path.join(plot_dir, "closure_scan_2d.png")
    fig.savefig(out_scan2d, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Closure/scan_2d": wandb.Image(out_scan2d)})

    # 2D histogram
    fig = plt.figure(figsize=(6, 5))
    xbins = np.geomspace(axis1_bkg[axis1_bkg > 0].min(), axis1_bkg.max(), 201)
    ybins = np.geomspace(axis2_bkg[axis2_bkg > 0].min(), axis2_bkg.max(), 201)
    plt.hist2d(axis1_bkg, axis2_bkg, bins=[xbins, ybins], norm=LogNorm(vmin=1), cmin=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
    plt.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("AE reco loss")
    plt.ylabel("Contrastive score (MD)")
    plt.title("AE vs Contrastive (bkg only)")
    plt.colorbar(label="Counts")
    out = os.path.join(plot_dir, "hist2d_bkg.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    wandb.log({"Hists2D/bkg": wandb.Image(out)})

    # per-class scatter + individual hist2d
    class_names  = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
    class_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}

    # combined scatter coloured by class (+ signal overlay)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() == 0:
            continue
        ax.scatter(axis1_bkg[m], axis2_bkg[m], s=0.3, alpha=0.15,
                   color=class_colors[cls], label=name, rasterized=True)
    for sig in signals:
        ax.scatter(sig["axis1"], sig["axis2"], s=0.5, alpha=0.4,
                   color=sig["color"], label=sig["label"], rasterized=True)
    ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("AE reco loss", fontsize=fs)
    ax.set_ylabel("Contrastive score (MD)", fontsize=fs)
    ax.set_title("AE vs Contrastive — all classes")
    ax.legend(markerscale=10, fontsize=fs_legend)
    out_combined = os.path.join(plot_dir, "hist2d_by_class_combined.png")
    fig.savefig(out_combined, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Hists2D/by_class_combined": wandb.Image(out_combined)})

    # individual hist2d per signal
    for sig in signals:
        fig = plt.figure(figsize=(6, 5))
        xbins_s = np.geomspace(sig["axis1"][sig["axis1"] > 0].min(), sig["axis1"].max(), 101)
        ybins_s = np.geomspace(sig["axis2"][sig["axis2"] > 0].min(), sig["axis2"].max(), 101)
        plt.hist2d(sig["axis1"], sig["axis2"], bins=[xbins_s, ybins_s], norm=LogNorm(vmin=1), cmin=1)
        plt.xscale("log"); plt.yscale("log")
        plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        plt.xlabel("AE reco loss", fontsize=fs)
        plt.ylabel("Contrastive score (MD)", fontsize=fs)
        plt.title(f"AE vs Contrastive — {sig['label']} (signal)")
        plt.colorbar(label="Counts")
        out_sig = os.path.join(plot_dir, f"hist2d_{sig['label']}.png")
        plt.savefig(out_sig, dpi=200, bbox_inches="tight")
        plt.close()
        wandb.log({f"Hists2D/{sig['label']}": wandb.Image(out_sig)})

    # individual hist2d per class
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() < 2:
            continue
        x_cls = axis1_bkg[m]
        y_cls = axis2_bkg[m]
        fig = plt.figure(figsize=(6, 5))
        xbins_c = np.geomspace(x_cls[x_cls > 0].min(), x_cls.max(), 101)
        ybins_c = np.geomspace(y_cls[y_cls > 0].min(), y_cls.max(), 101)
        plt.hist2d(x_cls, y_cls, bins=[xbins_c, ybins_c], norm=LogNorm(vmin=1), cmin=1)
        plt.xscale("log"); plt.yscale("log")
        plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        plt.xlabel("AE reco loss", fontsize=fs)
        plt.ylabel("Contrastive score (MD)", fontsize=fs)
        plt.title(f"AE vs Contrastive — {name}")
        plt.colorbar(label="Counts")
        out_cls = os.path.join(plot_dir, f"hist2d_{name}.png")
        plt.savefig(out_cls, dpi=200, bbox_inches="tight")
        plt.close()
        wandb.log({f"Hists2D/{name}": wandb.Image(out_cls)})

    # ── AE vs PCA-MD: scatter + KDE contour (skipped when --raw_md_axis2) ──
    if not config.get("raw_md_axis2"):
        fig, ax = plt.subplots(figsize=fig_size)
        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() == 0:
                continue
            ax.scatter(axis1_bkg[m], axis2_pca[m],
                       s=0.3, alpha=0.15, color=class_colors[cls], label=name, rasterized=True)
        for sig in signals:
            ax.scatter(sig["axis1"], sig["axis2_pca"], s=0.5, alpha=0.4,
                       color=sig["color"], label=sig["label"], rasterized=True)
        ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("AE reco loss", fontsize=fs)
        ax.set_ylabel("Contrastive score (PCA-MD)", fontsize=fs)
        ax.set_title("AE vs PCA-MD — all classes (scatter)")
        ax.legend(markerscale=10, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_pca_md_scatter = os.path.join(plot_dir, "hist2d_pca_md_scatter.png")
        fig.savefig(out_pca_md_scatter, dpi=200, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"Hists2D/pca_md_scatter": wandb.Image(out_pca_md_scatter)})

        # KDE contours in log(AE) vs log(PCA-MD) space
        rng_pca = np.random.default_rng(42)
        fig, ax = plt.subplots(figsize=fig_size)
        kde_legend_handles = []

        # compute global log-space range across all classes + signals
        all_lx, all_ly = [], []
        for cls, name in class_names.items():
            x_raw = axis1_bkg[labels_masked == cls]
            y_raw = axis2_pca[labels_masked == cls]
            valid = (x_raw > 0) & (y_raw > 0) & np.isfinite(x_raw) & np.isfinite(y_raw)
            if valid.sum() >= 50:
                all_lx.append(np.log10(x_raw[valid]))
                all_ly.append(np.log10(y_raw[valid]))
        for sig in signals:
            x_raw, y_raw = sig["axis1"], sig["axis2_pca"]
            valid = (x_raw > 0) & (y_raw > 0) & np.isfinite(x_raw) & np.isfinite(y_raw)
            if valid.sum() >= 50:
                all_lx.append(np.log10(x_raw[valid]))
                all_ly.append(np.log10(y_raw[valid]))
        glx_min, glx_max = np.concatenate(all_lx).min(), np.concatenate(all_lx).max()
        gly_min, gly_max = np.concatenate(all_ly).min(), np.concatenate(all_ly).max()
        xi_global, yi_global = np.mgrid[glx_min:glx_max:200j, gly_min:gly_max:200j]

        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() < 50:
                continue
            x_raw, y_raw = axis1_bkg[m], axis2_pca[m]
            color = class_colors[cls]
            valid = (x_raw > 0) & (y_raw > 0) & np.isfinite(x_raw) & np.isfinite(y_raw)
            lx = np.log10(x_raw[valid])
            ly = np.log10(y_raw[valid])
            if lx.shape[0] > 20_000:
                idx = rng_pca.choice(lx.shape[0], 20_000, replace=False)
                lx, ly = lx[idx], ly[idx]
            kde = gaussian_kde(np.vstack([lx, ly]))
            zi = kde(np.vstack([xi_global.flatten(), yi_global.flatten()]))
            zi_grid = zi.reshape(xi_global.shape)
            levels = zi_grid.max() * np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.88])
            ax.contour(10**xi_global, 10**yi_global, zi_grid,
                       levels=levels, colors=color, alpha=0.7, linewidths=1.5)
            kde_legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=name))

        for sig in signals:
            x_raw, y_raw = sig["axis1"], sig["axis2_pca"]
            color = sig["color"]
            valid = (x_raw > 0) & (y_raw > 0) & np.isfinite(x_raw) & np.isfinite(y_raw)
            lx = np.log10(x_raw[valid])
            ly = np.log10(y_raw[valid])
            if lx.shape[0] > 20_000:
                idx = rng_pca.choice(lx.shape[0], 20_000, replace=False)
                lx, ly = lx[idx], ly[idx]
            if lx.shape[0] < 50:
                continue
            kde = gaussian_kde(np.vstack([lx, ly]))
            zi = kde(np.vstack([xi_global.flatten(), yi_global.flatten()]))
            zi_grid = zi.reshape(xi_global.shape)
            levels = zi_grid.max() * np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.88])
            ax.contour(10**xi_global, 10**yi_global, zi_grid,
                       levels=levels, colors=color, alpha=0.7, linewidths=1.5)
            kde_legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=sig["label"]))

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("AE reco loss", fontsize=fs)
        ax.set_ylabel("Contrastive score (PCA-MD)", fontsize=fs)
        ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("AE vs PCA-MD — KDE contours")
        ax.legend(handles=kde_legend_handles, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        ax.grid(alpha=0.3)
        out_pca_md_kde = os.path.join(plot_dir, "hist2d_pca_md_kde.png")
        fig.savefig(out_pca_md_kde, dpi=200, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"Hists2D/pca_md_kde": wandb.Image(out_pca_md_kde)})

    # ── PCA scatter + KDE contour of encoder embeddings (skipped with --skip_embedding_pca) ──
    if not config.get("skip_embedding_pca"):
        pca = PCA(n_components=2)
        pca.fit(embeddings_masked[labels_masked == 1])
        emb_2d = pca.transform(embeddings_masked)

        for sig in signals:
            sig["emb_2d"] = pca.transform(sig["embeddings"])

        # PCA scatter
        fig, ax = plt.subplots(figsize=fig_size)
        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() == 0:
                continue
            ax.scatter(emb_2d[m, 0], emb_2d[m, 1],
                       s=0.5, alpha=0.12, color=class_colors[cls],
                       label=name, rasterized=True)
        for sig in signals:
            ax.scatter(sig["emb_2d"][:, 0], sig["emb_2d"][:, 1],
                       s=0.5, alpha=0.4, color=sig["color"],
                       label=sig["label"], rasterized=True)
        ax.set_xlabel("PCA Component 1", fontsize=fs)
        ax.set_ylabel("PCA Component 2", fontsize=fs)
        ax.set_title("Encoder embeddings — PCA (fit on QCD)")
        ax.legend(markerscale=10, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_pca_scatter = os.path.join(plot_dir, "pca_scatter_embeddings.png")
        fig.savefig(out_pca_scatter, dpi=200, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"PCA/scatter": wandb.Image(out_pca_scatter)})

        # KDE contour in PCA space
        rng = np.random.default_rng(42)
        fig, ax = plt.subplots(figsize=fig_size)
        legend_handles = []

        # global range across all classes + signals
        all_emb_x = [emb_2d[labels_masked == cls, 0] for cls in class_names if (labels_masked == cls).sum() >= 50]
        all_emb_y = [emb_2d[labels_masked == cls, 1] for cls in class_names if (labels_masked == cls).sum() >= 50]
        for sig in signals:
            all_emb_x.append(sig["emb_2d"][:, 0])
            all_emb_y.append(sig["emb_2d"][:, 1])
        gx_min, gx_max = np.concatenate(all_emb_x).min(), np.concatenate(all_emb_x).max()
        gy_min, gy_max = np.concatenate(all_emb_y).min(), np.concatenate(all_emb_y).max()
        xi_g, yi_g = np.mgrid[gx_min:gx_max:200j, gy_min:gy_max:200j]

        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() < 50:
                continue
            x, y = emb_2d[m, 0], emb_2d[m, 1]
            if x.shape[0] > 20_000:
                idx = rng.choice(x.shape[0], 20_000, replace=False)
                x, y = x[idx], y[idx]
            kde = gaussian_kde(np.vstack([x, y]))
            zi = kde(np.vstack([xi_g.flatten(), yi_g.flatten()]))
            zi_grid = zi.reshape(xi_g.shape)
            levels = zi_grid.max() * np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.88])
            ax.contour(xi_g, yi_g, zi_grid,
                       levels=levels, colors=class_colors[cls], alpha=0.7, linewidths=1.5)
            legend_handles.append(
                Line2D([0], [0], color=class_colors[cls], linewidth=1.5, label=name)
            )

        for sig in signals:
            xs, ys = sig["emb_2d"][:, 0], sig["emb_2d"][:, 1]
            if xs.shape[0] > 20_000:
                idx = rng.choice(xs.shape[0], 20_000, replace=False)
                xs, ys = xs[idx], ys[idx]
            if xs.shape[0] < 50:
                continue
            kde_s = gaussian_kde(np.vstack([xs, ys]))
            zi_s = kde_s(np.vstack([xi_g.flatten(), yi_g.flatten()]))
            zi_s_grid = zi_s.reshape(xi_g.shape)
            levels_s = zi_s_grid.max() * np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.88])
            ax.contour(xi_g, yi_g, zi_s_grid,
                       levels=levels_s, colors=sig["color"], alpha=0.7, linewidths=1.5)
            legend_handles.append(
                Line2D([0], [0], color=sig["color"], linewidth=1.5, label=sig["label"])
            )

        ax.set_xlabel("PCA Component 1", fontsize=fs)
        ax.set_ylabel("PCA Component 2", fontsize=fs)
        ax.set_title("KDE contours — encoder embeddings PCA (fit on QCD)")
        ax.legend(handles=legend_handles, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        ax.grid(alpha=0.3)
        out_pca_kde = os.path.join(plot_dir, "pca_kde_embeddings.png")
        fig.savefig(out_pca_kde, dpi=200, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"PCA/kde_contour": wandb.Image(out_pca_kde)})

    # ── Corner plot: all pairwise PCA-MD components ──
    if emb_pca is not None and n_pca >= 2:
        pairs = [(i, j) for i in range(n_pca) for j in range(i + 1, n_pca)]
        n_pairs = len(pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]
        for ax, (ci, cj) in zip(axes, pairs):
            for cls, name in class_names.items():
                m = labels_masked == cls
                if m.sum() == 0:
                    continue
                ax.scatter(emb_pca[m, ci], emb_pca[m, cj],
                           s=0.3, alpha=0.12, color=class_colors[cls],
                           label=name, rasterized=True)
            for sig in signals:
                ax.scatter(sig["emb_pca"][:, ci], sig["emb_pca"][:, cj],
                           s=0.5, alpha=0.4, color=sig["color"],
                           label=sig["label"], rasterized=True)
            ax.set_xlabel(f"PCA Component {ci + 1}", fontsize=fs)
            ax.set_ylabel(f"PCA Component {cj + 1}", fontsize=fs)
            ax.legend(markerscale=10, fontsize=fs_legend)
            ax.tick_params(axis="both", labelsize=fs_leg)
        fig.suptitle("PCA-MD space — pairwise components (fit on QCD)", fontsize=fs)
        plt.tight_layout()
        out_corner = os.path.join(plot_dir, "pca_corner.png")
        fig.savefig(out_corner, dpi=200, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"PCA/corner": wandb.Image(out_corner)})

    # profile: <AE> vs contrastive
    fig, ax = plt.subplots(figsize=fig_size)
    profile_plot(ax, axis2_bkg, axis1_bkg, nbins=60, logx=True)
    ax.set_xlabel("Contrastive score (MD)", fontsize=fs)
    ax.set_ylabel("Mean AE reco loss",      fontsize=fs)
    ax.set_title("⟨AE loss⟩ vs contrastive MD")
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    p1_path = os.path.join(plot_dir, "profile_AE_vs_contrastive.png")
    fig.savefig(p1_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/AE_vs_contrastive": wandb.Image(p1_path)})

    # profile: <contrastive> vs AE
    fig, ax = plt.subplots(figsize=fig_size)
    profile_plot(ax, axis1_bkg, axis2_bkg, nbins=60, logx=True)
    ax.set_xlabel("AE reco loss",                  fontsize=fs)
    ax.set_ylabel("Mean contrastive score (MD)",   fontsize=fs)
    ax.set_title("⟨contrastive MD⟩ vs AE loss")
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    p2_path = os.path.join(plot_dir, "profile_contrastive_vs_AE.png")
    fig.savefig(p2_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/contrastive_vs_AE": wandb.Image(p2_path)})

    # per-class profile: <AE> vs contrastive MD
    fig, ax = plt.subplots(figsize=fig_size)
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() < 20:
            continue
        profile_plot(ax, axis2_bkg[m], axis1_bkg[m], nbins=40, logx=True, label=name)
    ax.set_xlabel("Contrastive score (MD)", fontsize=fs)
    ax.set_ylabel("Mean AE reco loss",      fontsize=fs)
    ax.set_title("⟨AE loss⟩ vs contrastive MD (by class)")
    ax.legend(fontsize=fs_legend)
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    p3_path = os.path.join(plot_dir, "profile_AE_vs_contrastive_by_class.png")
    fig.savefig(p3_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/AE_vs_contrastive_by_class": wandb.Image(p3_path)})

    # per-class profile: <contrastive MD> vs AE
    fig, ax = plt.subplots(figsize=fig_size)
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() < 20:
            continue
        profile_plot(ax, axis1_bkg[m], axis2_bkg[m], nbins=40, logx=True, label=name)
    ax.set_xlabel("AE reco loss",                  fontsize=fs)
    ax.set_ylabel("Mean contrastive score (MD)",   fontsize=fs)
    ax.set_title("⟨contrastive MD⟩ vs AE loss (by class)")
    ax.legend(fontsize=fs_legend)
    plt.tick_params(axis='x', labelsize=fs_leg)
    plt.tick_params(axis='y', labelsize=fs_leg)
    p4_path = os.path.join(plot_dir, "profile_contrastive_vs_AE_by_class.png")
    fig.savefig(p4_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Profiles/contrastive_vs_AE_by_class": wandb.Image(p4_path)})

    # 1D scan for closure + S/sqrt(B) - wider/looser range than the `percent` grid used
    # for the ABCD working-point search above, so this diagnostic curve reaches into
    # loose-cut territory (small p -> most events pass -> high efficiency) instead of
    # stopping at p=0.75.
    percent_plot = np.linspace(0.10, 0.9995, 80)
    effs, closure_ratio, closure_unc, s_over_sqrtb = [], [], [], []
    scan_t1, scan_t2 = [], []
    Ntot_bkg = float(len(axis1_qcd))

    for p in percent_plot:
        t1, t2, A, B, C, D = abcd_counts(axis1_qcd, axis2_qcd, p, p)
        A_hat  = (B * C) / max(D, 1e-8)
        ratio  = A_hat / max(A, 1e-8)
        invA   = 0.0 if A == 0 else 1.0 / A
        invB   = 0.0 if B == 0 else 1.0 / B
        invC   = 0.0 if C == 0 else 1.0 / C
        invD   = 0.0 if D == 0 else 1.0 / D
        rel_var = invA + invB + invC + invD
        sigma  = abs(ratio) * np.sqrt(rel_var) if rel_var > 0 else 0.0

        effs.append(A / max(Ntot_bkg, 1.0))
        closure_ratio.append(ratio)
        closure_unc.append(sigma)
        s_over_sqrtb.append(0.0 / np.sqrt(max(A, 1e-8)))  # no signal yet
        scan_t1.append(t1)
        scan_t2.append(t2)

    effs          = np.array(effs)
    closure_ratio = np.array(closure_ratio)
    closure_unc   = np.array(closure_unc)
    s_over_sqrtb  = np.array(s_over_sqrtb)
    scan_t1       = np.array(scan_t1)
    scan_t2       = np.array(scan_t2)

    order         = np.argsort(effs)
    effs          = effs[order]
    closure_ratio = closure_ratio[order]
    closure_unc   = closure_unc[order]
    s_over_sqrtb  = s_over_sqrtb[order]
    scan_t1       = scan_t1[order]
    scan_t2       = scan_t2[order]

    eff_opt   = best["A"] / max(Ntot_bkg, 1.0)
    ratio_opt = best["A_hat"] / max(best["A"], 1e-8)

    # tightest cut still within ±5% closure band
    closure_tol = config.get("closure_tolerance", 0.10)
    within_band = np.abs(closure_ratio - 1.0) <= closure_tol
    tightest = {}
    if within_band.any():
        idx = int(np.where(within_band)[0].min())   # smallest eff = tightest cut
        tightest = {
            "eff":   float(effs[idx]),
            "ratio": float(closure_ratio[idx]),
            "t1":    float(scan_t1[idx]),
            "t2":    float(scan_t2[idx]),
        }
        print(f"Tightest good cut (|closure|<={closure_tol*100:.0f}%): "
              f"eff={tightest['eff']:.2e}, t1={tightest['t1']:.4g}, t2={tightest['t2']:.4g}", flush=True)
        wandb.log({
            "Closure/tightest_eff":  tightest["eff"],
            "Closure/tightest_t1":   tightest["t1"],
            "Closure/tightest_t2":   tightest["t2"],
            "Closure/tightest_ratio": tightest["ratio"],
        })

    # closure plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(effs, closure_ratio, c="g", label="AE + Contrastive (MD)")
    ax.fill_between(effs,
                    closure_ratio - closure_unc,
                    closure_ratio + closure_unc,
                    facecolor="g", alpha=0.5, interpolate=True)
    ax.plot(effs, np.ones_like(effs),          linestyle="-",  color="black")
    ax.plot(effs, np.full_like(effs, 0.95),    linestyle="--", color="black")
    ax.plot(effs, np.full_like(effs, 1.05),    linestyle="--", color="black")
    ax.plot([eff_opt], [ratio_opt], marker="o", c="red", label="Optimized")
    if tightest:
        ax.axvline(tightest["eff"], color="blue", linestyle=":", linewidth=1.5)
        ax.plot([tightest["eff"]], [tightest["ratio"]], marker="o", c="blue",
                label=f"Tightest (±10%): t1={tightest['t1']:.3g}, t2={tightest['t2']:.3g}")
    ax.set_xlabel("Selection Efficiency (bkg A/Ntot)", fontsize=fs)
    ax.set_ylabel("Predicted Bkg. / True Bkg.",        fontsize=fs)
    ax.set_ylim([0.0, 1.5])
    ax.set_xscale("log")
    plt.tick_params(axis="x", labelsize=fs_leg)
    plt.tick_params(axis="y", labelsize=fs_leg)
    plt.legend(loc="lower right", fontsize=fs_legend)
    closure_path = os.path.join(plot_dir, "cut_and_count_bkg_check.png")
    plt.savefig(closure_path, dpi=200, bbox_inches="tight")
    plt.close()
    wandb.log({"Closure/plot": wandb.Image(closure_path)})

    # S/sqrt(B) placeholder
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(effs, s_over_sqrtb, color="red", label=r"$S/\sqrt{B}$")
    ax.plot([eff_opt], [0.0], marker="o", color="black")
    ax.set_xlabel("Selection Efficiency (bkg A/Ntot)", fontsize=fs)
    ax.set_ylabel(r"$S/\sqrt{B}$",                     fontsize=fs)
    ax.set_xscale("log")
    plt.tick_params(axis="x", labelsize=fs_leg)
    plt.tick_params(axis="y", labelsize=fs_leg)
    plt.legend(loc="best", fontsize=fs_legend)
    sig_path = os.path.join(plot_dir, "s_over_sqrtb_vs_bkg_eff.png")
    plt.savefig(sig_path, dpi=200, bbox_inches="tight")
    plt.close()
    wandb.log({"Signal/s_over_sqrtb_vs_bkg_eff": wandb.Image(sig_path)})

    # 2D scatter coloured by class with tightest-closure thresholds
    if tightest:
        fig, ax = plt.subplots(figsize=(6, 5))
        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() == 0:
                continue
            ax.scatter(axis1_bkg[m], axis2_bkg[m], s=0.3, alpha=0.15,
                       color=class_colors[cls], label=name, rasterized=True)
        for sig in signals:
            ax.scatter(sig["axis1"], sig["axis2"], s=0.5, alpha=0.4,
                       color=sig["color"], label=sig["label"], rasterized=True)
        ax.axvline(tightest["t1"], color="blue",  linestyle="--", linewidth=1.5, label="Tightest")
        ax.axhline(tightest["t2"], color="blue",  linestyle="--", linewidth=1.5)
        ax.axvline(t1_opt,         color="black", linestyle=":",  linewidth=1.0, label="Optimized")
        ax.axhline(t2_opt,         color="black", linestyle=":",  linewidth=1.0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("AE reco loss", fontsize=fs)
        ax.set_ylabel("Contrastive score (MD)", fontsize=fs)
        ax.set_title("AE vs Contrastive — tightest closure cut (±10%)")
        ax.legend(markerscale=10, fontsize=fs_legend, loc="lower right")
        out_tightest = os.path.join(plot_dir, "hist2d_tightest_cut.png")
        fig.savefig(out_tightest, dpi=200, bbox_inches="tight")
        plt.close(fig)
        wandb.log({"Hists2D/tightest_cut": wandb.Image(out_tightest)})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast_ckpt",      required=True)
    parser.add_argument("--contrast_test_pt",   required=True,
                        help="Path to SM cocktail test .pt (DY/QCD/TT/WJets). "
                             "If --qcd_pt is given, the QCD events here are replaced.")
    parser.add_argument("--qcd_pt",             default=None,
                        help="Path to a separate QCD-only .pt file (e.g. mquinna test slice). "
                             "Its events replace QCD (label==1) from --contrast_test_pt, "
                             "so DY/TT/WJets still come from the smcocktail test file.")
    parser.add_argument("--ae_scores_bkg_test", default=None,
                        help="Pre-computed AE scores .pt file. Not needed if checkpoint contains a joint AE.")
    parser.add_argument("--signal_pt", nargs="*", default=None,
                        help="Signal .pt file(s). Pass one or more paths.")
    parser.add_argument("--signal_label", nargs="*", default=None,
                        help="Label(s) for signal sample(s). Must match --signal_pt count.")
    parser.add_argument("--outdir",             default="outputs_abcd")
    parser.add_argument("--min_A", type=int,    default=200)
    parser.add_argument("--min_D", type=int,    default=1000)
    parser.add_argument("--n_pca", type=int,    default=None,
                        help="Override pca_n_components from checkpoint. Use 6 for full-dim PCA.")
    parser.add_argument("--min_md",            action="store_true",
                        help="Axis 2 = min(MD_DY, MD_QCD, MD_WJets) instead of QCD-only MD.")
    parser.add_argument("--raw_md_axis2",      action="store_true",
                        help="Use raw Mahalanobis distance as axis 2 instead of PCA-MD.")
    parser.add_argument("--skip_embedding_pca", action="store_true",
                        help="Skip the 2D PCA embedding scatter/KDE visualization plots.")
    parser.add_argument("--wandb_run_name", default=None,
                        help="WandB run name. Defaults to WandB auto-generated name.")
    args = parser.parse_args()

    ABCD(vars(args))
