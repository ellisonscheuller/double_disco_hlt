import torch
import torch.nn as nn
import datetime
import os
import logging
import argparse
import importlib
import wandb
import numpy as np
from embedding_pca_per_batch.models import TransformerEncoder, Projector
from embedding_pca_epoch.autoencoder import Autoencoder
from embedding_pca_per_batch.training import make_train_val_split, build_train_val_loaders, train_epoch, validate_epoch, EarlyStopping, cosine_schedule_with_warmup, cosine_constrastive_schedule, linear_warmup_weight
from embedding_pca_epoch.utils.data_utils import compute_normalization_constants
from embedding_pca_epoch.utils.cfg_handler import train_config, data_config
from embedding_pca_epoch.utils.data_utils import compute_class_weights

# Use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Set up logging: messages go both to stdout and to a timestamped file in logs/
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JEPA")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/training_{timestamp}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def main(data_path: str, cfg: train_config, cfg_data: data_config, test_mode: bool = False):

    # ── WandB init ─────────────────────────────────────────────────────────────
    # Dump the full config so every run is fully reproducible from the wandb page.
    # In sweep mode, wandb.config will override cfg values below via cfg.hp().
    run = wandb.init(
        project="embedding_hlt",
        config = {
            **cfg.get_entire_cfg(),
            **cfg_data.get_entire_cfg()
        },
    )
    is_sweep = cfg.is_sweep()
    if is_sweep:
        logger.info("Running in sweep mode with wandb.config overrides.")
    if not is_sweep:
        # Give the run a human-readable name so it's easy to find in the wandb UI
        run.name = f"{cfg.get_model_name()}_{timestamp}"
        logger.info(f"Running in standard mode using config file values. Run name: {run.name}")

    # ── Hyperparameters ────────────────────────────────────────────────────────
    # cfg.hp(name, default) returns the wandb sweep value if running a sweep,
    # otherwise the value from the config yaml, otherwise the default.
    num_epochs = cfg.hp("num_epochs", 400 if not is_sweep else 50)
    patience = cfg.hp("early_stopping_patience", 100 if not is_sweep else 20)
    val_split = cfg.get_trdata_cfg("val_split", 0.1)
    pairwise = cfg.get_trdata_cfg("pairwise", False)
    class_weights_setting = cfg_data.get("class_weights", None)
    pfcands = cfg_data.get("pfcands", True)          # whether input is PF candidates (vs. object-level)
    contrast_loss = cfg.hp("contrast_loss", "InfoNCELoss")
    preproc_type = cfg.get_trdata_cfg("preproc_type", "PFPreProcessor")
    mixed_prec = cfg.hp("mixed_prec", False)         # AMP (FP16) flag

    # Sweepable architecture / training HPs
    num_heads = cfg.hp("num_heads", 8)               # transformer attention heads
    num_layers = cfg.hp("num_layers", 4)             # transformer depth
    embed_size = cfg.hp("embed_size", 128)           # per-token embedding dimension
    latent_dim = cfg.hp("latent_dim", 6)             # encoder output (event-level) dimension
    proj_dim = cfg.hp("proj_dim", 12)                # projector output dimension for contrastive loss
    linear_dim = cfg.hp("linear_dim", None)          # if set, switches encoder to Linformer
    dim_feedforward = cfg.hp("dim_feedforward", 2048) # FFN hidden size inside each transformer block
    contrast_temp = cfg.hp("contrast_temp", 0.07)   # InfoNCE temperature
    contrastive_weight = cfg.hp("contrastive_weight", 0.05) # Min (or fixed) weight on contrastive loss
    contrastive_max = cfg.hp("contrastive_max", None) # if set, ramp contrastive weight up to this value
    contrastive_warmup = cfg.hp("contrastive_warmup", 0.05) # fraction of total steps for contrastive ramp
    lr = cfg.hp("lr", 1e-3)
    lr_min = cfg.hp("lr_min", 0.0)                  # floor for cosine LR decay
    lr_warmup = cfg.hp("lr_warmup", 0.05)           # fraction of total steps used for linear LR warmup
    batch_size = cfg.hp("batch_size", 256)

    # GradScaler is a no-op when enabled=False, so this is safe to always create
    logger.info("Scaler for mixed precision training: {}".format(mixed_prec))
    scaler = torch.cuda.amp.GradScaler(enabled=((device=="cuda") and mixed_prec))

    # ── Data loading ───────────────────────────────────────────────────────────
    # In test_mode, only use 10% of data so the training loop runs quickly for debugging
    if test_mode:
        num_events = int(0.10 * cfg_data.get("nevents_per_class") * cfg_data.get_file_label_map().__len__())
        logger.info("Test mode enabled: using only 10% of the data for training and validation.")
        logger.info(f"Number of events: {num_events} (10% of total)")

    # Load file once — extract PF candidates and obj features in one shot to avoid double RAM peak
    _n = num_events if test_mode else -1
    _raw = torch.load(data_path, map_location="cpu")
    if isinstance(_raw, dict):
        feature_block = _raw['pf'][:_n] if _n > 0 else _raw['pf']
        label_block   = (_raw['label'][:_n] if _n > 0 else _raw['label']).long()
        feature_block = torch.nan_to_num(feature_block, nan=0.0, posinf=0.0, neginf=0.0)
        _obj_raw = (_raw["obj"][:_n] if _n > 0 else _raw["obj"]) if "obj" in _raw else None
    else:
        from embedding_pca_epoch.utils.data_utils import clean_data
        data = _raw[:_n] if _n > 0 else _raw
        feature_block, label_block = clean_data(data)
        _obj_raw = None
    del _raw

    # Split by index so we can apply the same split to the obj-level features below
    X_tr, y_tr, X_val, y_val, idx_tr, idx_val = make_train_val_split(feature_block, label_block, val_size=val_split)
    assert X_tr.device.type == "cpu"
    assert y_tr.device.type == "cpu"

    # ── AE branch (axis 1 of double DisCo) ────────────────────────────────────
    # The AE operates on object-level features (pt, η, φ, type_id for each PF candidate,
    # all concatenated into a flat vector). It is only built and trained when disco_weight > 0.
    # When disco_weight == 0, this whole block is skipped and ae_model stays None.
    disco_weight    = cfg.hp("disco_weight",    0.0)  # weight on the DisCo decorrelation term
    closure_weight  = cfg.hp("closure_weight",  0.0)  # weight on the ABCD closure loss
    ae_reco_weight  = cfg.hp("ae_reco_weight",  1.0)  # weight on the AE reconstruction loss
    disco_warmup    = cfg.hp("disco_warmup",    0.0)  # fraction of total steps to ramp disco from 0
    closure_warmup  = cfg.hp("closure_warmup",  0.0)  # fraction of total steps to ramp closure from 0
    obj_tr = obj_val = None
    ae_model = None
    if disco_weight > 0.0 or closure_weight > 0.0:
        obj_all = _obj_raw
        # Take 4 features per PF candidate and flatten: shape (N, n_cands*4)
        # Same preprocessing as train_ae_axis1.py to keep inference consistent
        obj_flat = obj_all[:, :, :4].reshape(obj_all.shape[0], -1).float().numpy()
        # Z-score normalise using training-set statistics
        mu  = obj_flat.mean(axis=0).astype(np.float32)
        std = obj_flat.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std)        # avoid division by zero for constant features
        obj_norm = torch.from_numpy((obj_flat - mu) / (std + 1e-8))
        del obj_all, obj_flat
        # Apply the same train/val split as the PF candidate data
        obj_tr  = obj_norm[idx_tr]
        obj_val = obj_norm[idx_val]
        # Save scaler so eval_abcd.py can apply the exact same normalisation at inference
        obj_scaler = {"mu": torch.from_numpy(mu), "std": torch.from_numpy(std)}
        logger.info(f"Loaded object-level features for AE: shape {obj_norm.shape}")

        ae_feat = obj_norm.shape[1]
        del obj_norm
        ae_latent  = cfg.hp("ae_latent",  16)
        ae_enc     = cfg.hp("ae_enc_nodes", [512, 256])
        ae_dec     = cfg.hp("ae_dec_nodes", [256, 512])
        ae_model = Autoencoder({
            "features":       ae_feat,
            "latent_dim":     ae_latent,
            "encoder_config": {"nodes": ae_enc},
            "decoder_config": {"nodes": ae_dec + [ae_feat]},  # decoder mirrors encoder + output layer
            "alpha": 1.0,
        }).to(device)
        logger.info(f"Built AE: input={ae_feat}, latent={ae_latent}, enc={ae_enc}, dec={ae_dec}")

    # ── Class balance logging ──────────────────────────────────────────────────
    # Useful for spotting severe imbalance that might need class weighting
    num_classes = int(label_block.max().item()) + 1
    class_count_tr = torch.bincount(y_tr, minlength=num_classes)
    logger.info("Class counts in training set:")
    for i in range(num_classes):
        logger.info(f"  Class {i}: {class_count_tr[i].item()} events")
    class_count_val = torch.bincount(y_val, minlength=num_classes)
    logger.info("Class counts in validation set:")
    for i in range(num_classes):
        logger.info(f"  Class {i}: {class_count_val[i].item()} events")

    # ── DataLoaders ────────────────────────────────────────────────────────────
    # Normalisation constants computed from training set only — never touch val stats
    norm_constants = compute_normalization_constants(X_tr) if not pfcands else {}
    train_loader, val_loader = build_train_val_loaders(
        X_tr, y_tr, X_val, y_val, device=device, batch_size=batch_size, pfcands=pfcands,
        obj_tr=obj_tr, obj_val=obj_val,
    )

    # ── Model construction ─────────────────────────────────────────────────────
    # preproc: handles any input-level transformations (e.g. log-scaling pt)
    preproc_class = getattr(importlib.import_module("embedding_pca_epoch.preprocs"), preproc_type)
    preproc = preproc_class(norm_constants).to(device)

    # encoder (axis 2): Transformer over PF candidates → single event-level latent vector
    # If linear_dim is set this becomes a Linformer (linear-complexity attention)
    encoder = TransformerEncoder(
        num_features=preproc.num_features,
        embed_size=embed_size,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        linear_dim=linear_dim,
        num_tokens=feature_block.size(1) if linear_dim is not None else None,
        pairwise=pairwise,
        pre_processor=preproc
    ).to(device).train()
    # projector: small MLP on top of encoder; only used during training for contrastive loss
    # (at inference we use the raw encoder output, not the projector output)
    projector = Projector(latent_dim, proj_dim, hidden_dim=(proj_dim*4)).to(device).train()
    # classifier: linear head for supervised cross-entropy loss
    classifier = nn.Linear(proj_dim, num_classes).to(device).train()

    # ── Loss functions ─────────────────────────────────────────────────────────
    # Class weights down-weight over-represented classes so the CE loss doesn't dominate on majority
    class_weights = compute_class_weights(label_block, setting=class_weights_setting).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    # Contrastive loss (InfoNCE by default): pulls same-class embeddings together, pushes different apart
    contrast_loss_class = getattr(importlib.import_module("embedding_pca_per_batch.loss"), contrast_loss)
    criterion = contrast_loss_class(temperature=contrast_temp)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Single Adam over everything — encoder, projector, classifier, and AE share one optimiser
    # so all branches update together each step
    ae_params = list(ae_model.parameters()) if ae_model is not None else []
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(projector.parameters()) +
        list(classifier.parameters()) +
        ae_params,
        lr=lr
    )

    # ── Learning rate scheduler ────────────────────────────────────────────────
    # Cosine decay with a short linear warmup at the start.
    # Warmup prevents large gradient steps before the model has settled.
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(lr_warmup * total_steps)
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
        lr=lr, # max present => scheduled, otherwise use lr_max as fixed min
        lr_min=lr_min
    )

    # ── Contrastive weight scheduler ──────────────────────────────────────────
    # Optional: start the contrastive loss weight low and ramp it up during training.
    # The idea is to let CE loss dominate early (so embeddings are roughly class-separated)
    # before the contrastive term pushes embeddings into a tighter geometry.
    if contrastive_max is not None:
        contrastive_warmup_steps = int(contrastive_warmup * total_steps)
        contrastive_schedule = cosine_constrastive_schedule(
            weight_min = contrastive_weight,
            weight_max = contrastive_max,
            warmup_steps = contrastive_warmup_steps,
            total_steps = total_steps
        )

    # ── DisCo / Closure weight schedulers ─────────────────────────────────────
    # Ramp from 0 so CE/contrastive can establish a good latent space first
    if disco_warmup > 0.0 and disco_weight > 0.0:
        disco_schedule = linear_warmup_weight(disco_weight, int(disco_warmup * total_steps))
    else:
        disco_schedule = disco_weight

    if closure_warmup > 0.0 and closure_weight > 0.0:
        closure_schedule = linear_warmup_weight(closure_weight, int(closure_warmup * total_steps))
    else:
        closure_schedule = closure_weight

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val = float("inf")
    es = EarlyStopping(patience=patience, mode="min", min_delta=0.0)

    # Checkpoint path is fixed at the start so the best model always overwrites the same file
    model_path = os.path.join(os.getcwd(), "checkpoints", f"{cfg.get_model_name()}_encoder_{timestamp}.pth")

    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        # Forward + backward pass over all training batches; updates model weights
        tr = train_epoch(
            encoder,
            projector,
            classifier,
            ce_loss_fn,
            criterion,
            train_loader,
            norm_constants,
            device,
            optimizer,
            scheduler=scheduler,
            # Pass the schedule object directly if ramping, otherwise a fixed scalar
            contrastive_weight=contrastive_weight if contrastive_max is None else contrastive_schedule,
            pairwise=pairwise,
            num_classes=num_classes,
            scaler=scaler,
            ae_model=ae_model,
            ae_reco_weight=ae_reco_weight,
            disco_weight=disco_schedule,
            closure_weight=closure_schedule,
        )
        # Evaluation pass — no gradients, no weight updates
        va = validate_epoch(
            encoder,
            projector,
            classifier,
            ce_loss_fn,
            criterion,
            val_loader,
            norm_constants,
            device,
            contrastive_weight=contrastive_weight if contrastive_max is None else contrastive_schedule,
            pairwise=pairwise,
            num_classes=num_classes,
            ae_model=ae_model,
            ae_reco_weight=ae_reco_weight,
        )

        log_str = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: loss {tr['loss']:.6f}, Contrast {tr['contrast']:.6f}, CE {tr['ce']:.6f}, AE {tr['ae_reco']:.6f}, DisCo {tr['disco']:.6f}, Closure {tr['closure']:.6f}, acc {tr['acc']:.4f} | "
            f"Val:   loss {va['loss']:.6f}, Contrast {va['contrast']:.6f}, CE {va['ce']:.6f}, AE {va['ae_reco']:.6f}, acc {va['acc']:.4f}"
        )
        logger.info(log_str)

        # Save a checkpoint whenever val loss improves — this keeps the best model seen so far.
        # The checkpoint includes everything needed to resume training or run inference.
        if va["loss"] < best_val:
            best_val = va["loss"]
            ckpt = {
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                # Move norm constants to CPU so the checkpoint is device-agnostic
                "norm_constants": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in norm_constants.items()},
            }
            if ae_model is not None:
                ckpt["ae"] = ae_model.state_dict()
                # Save the normalisation scaler so eval_abcd.py applies the same transform
                ckpt["ae_scaler"] = obj_scaler
            torch.save(ckpt, model_path)
            logger.info(f"Saved best checkpoint to: {model_path}")

        # Log all metrics + current LR and contrastive weight to wandb
        run.log(
            {
                "Train Loss": tr["loss"],
                "Train Contrastive": tr["contrast"],
                "Train CrossEntropy": tr["ce"],
                "Train AE Reco": tr["ae_reco"],
                "Train DisCo": tr["disco"],
                "Train Closure": tr["closure"],
                "Train Accuracy": tr["acc"],
                "Val Loss": va["loss"],
                "Val Contrastive": va["contrast"],
                "Val CrossEntropy": va["ce"],
                "Val AE Reco": va["ae_reco"],
                "Val Accuracy": va["acc"],
                # to track change in hps
                "Learning Rate": scheduler.get_last_lr()[0],
                "Contrastive Weight": contrastive_weight if contrastive_max is None else contrastive_schedule.get(),
                "DisCo Weight": disco_schedule if isinstance(disco_schedule, float) else disco_schedule.get(),
                "Closure Weight": closure_schedule if isinstance(closure_schedule, float) else closure_schedule.get(),
            }, step=epoch
        )

        # log lr and contrast weight
        current_lr = scheduler.get_last_lr()[0]
        run.log({"Learning Rate": current_lr}, step=epoch)
        if contrastive_max is not None:
            current_contrastive_weight = contrastive_schedule.get()
            run.log({"Contrastive Weight": current_contrastive_weight}, step=epoch)

        # Stop early if val loss hasn't improved for `patience` epochs
        if es.step(va["loss"]):
            logger.info("Early stopping triggered.")
            break
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True, help="Path to the data config .yaml file")
    parser.add_argument("--train_cfg", required=True, help="Path to the training config .yaml file")
    parser.add_argument("--data", required=True, help="Path to the input .pt file")
    parser.add_argument("--test_mode", action="store_true", help="If set, runs training with only 10 percent of the data.")
    args = parser.parse_args()

    tr_cfg = train_config(args.train_cfg)
    data_cfg = data_config(args.data_cfg)

    logger.info(f"Using train config file: {args.train_cfg}")
    logger.info(f"Entire train config: {tr_cfg.get_entire_cfg()}")

    logger.info(f"Using data processing config file: {args.data_cfg}")
    logger.info(f"Entire data processing config: {data_cfg.get_entire_cfg()}")

    main(args.data, tr_cfg, data_cfg, test_mode=args.test_mode)
