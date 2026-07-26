# monitoring.py
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from embedding_pca_epoch.dataloader import PFCandsDataset, PUPPIDataset
from embedding_pca_per_batch.loss import distance_corr
from embedding_pca_epoch.utils.data_utils import delta_r_from_normalized
from embedding_pca_epoch.utils.data_utils import EPS
from typing import Union

class EarlyStopping:
    """
    Simple early stopping on a monitored value (default: minimize 'loss').
    mode='min' or 'max'
    """
    def __init__(self, patience=20, mode="min", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0

    def step(self, value):
        import math
        if math.isnan(value) or math.isinf(value):
            return True  # stop immediately on NaN/Inf loss

        if self.best is None:
            self.best = value
            return False  # do not stop

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience

def make_train_val_split(features, y, val_size=0.10, random_state=42, y_are_labels=True):
    """
    Stratified split of event tensors into train/val.
    features: [E, N, F] (normalized)
    y (labels):   [E]
    """
    idx = torch.arange(features.shape[0])
    idx_tr, idx_val = train_test_split(
        idx.cpu().numpy(),
        test_size=val_size,
        random_state=random_state,
        stratify=y.cpu().numpy() if y_are_labels else None
    )
    idx_tr = torch.tensor(idx_tr, dtype=torch.long, device=features.device)
    idx_val = torch.tensor(idx_val, dtype=torch.long, device=features.device)

    X_tr = features.index_select(0, idx_tr)
    y_tr = y.index_select(0, idx_tr)
    X_val = features.index_select(0, idx_val)
    y_val = y.index_select(0, idx_val)
    return X_tr, y_tr, X_val, y_val, idx_tr, idx_val

def build_train_val_loaders(
    X_tr, y_tr, X_val, y_val, device, batch_size=2048, pfcands=False,
    obj_tr=None, obj_val=None,
):
    """
    Builds DataLoaders. IMPORTANT: pass TRAIN norm_constants to BOTH loaders.
    obj_tr/val: optional [N, D] tensors of normalised object-level features for the AE.
    """
    if pfcands:
        ds_tr  = PFCandsDataset(X_tr,  y_tr, device, obj_features=obj_tr)
        ds_val = PFCandsDataset(X_val, y_val, device, obj_features=obj_val)
    else:
        ds_tr  = PUPPIDataset(X_tr,  y_tr,  device=device)
        ds_val = PUPPIDataset(X_val, y_val, device=device)

    train_loader = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def sigmoid_counts(var1, var2, cut1, cut2, weights, scale=100.0):
    """Differentiable ABCD region counts using sigmoid soft-cuts (from DiSCoTEC paper)."""
    v1 = var1.view(-1)
    v2 = var2.view(-1)
    w  = weights.view(-1)

    s1_high = torch.sigmoid(scale * (v1 - cut1))
    s1_low  = torch.sigmoid(scale * (cut1 - v1))
    s2_high = torch.sigmoid(scale * (v2 - cut2))
    s2_low  = torch.sigmoid(scale * (cut2 - v2))

    NA = torch.sum(s1_high * s2_high * w)
    NB = torch.sum(s1_high * s2_low  * w)
    NC = torch.sum(s1_low  * s2_high * w)
    ND = torch.sum(s1_low  * s2_low  * w)
    return NA, NB, NC, ND


def closure_loss_batch(var1, var2, weights, symmetrize=True,
                       n_events_min=10, max_tries=20, scale=50.0, n_cuts=5,
                       cut_beta_alpha=3.0):
    """
    ABCD closure loss on a batch.
    Normalizes both variables to their 1-99% quantile range (→ [0,1]) before
    computing soft ABCD counts, so the sigmoid scale is meaningful regardless
    of the variables' absolute scale (important when var2 is Mahalanobis distance).
    Averages over n_cuts random cuts to reduce gradient variance.
    Cuts are drawn from Beta(cut_beta_alpha, 1), which skews toward 1 (mean
    0.75 at alpha=3) so training probes the same high-percentile tail region
    that eval_abcd.py's working-point scan uses (percentiles ~0.75-0.9995),
    instead of wasting most steps on loose cuts that don't affect the ABCD
    region actually reported at eval time.
    Returns mean of |NA*ND - NB*NC| / (NA*ND + NB*NC) over valid cuts.
    """
    v1 = var1.view(-1)
    v2 = var2.view(-1)
    w  = weights.view(-1)

    with torch.no_grad():
        x_min = torch.quantile(v1, 0.01).item()
        x_max = torch.quantile(v1, 0.99).item()
        y_min = torch.quantile(v2, 0.01).item()
        y_max = torch.quantile(v2, 0.99).item()

    x_range = x_max - x_min + 1e-8
    y_range = y_max - y_min + 1e-8

    # Normalize to [0, 1] so sigmoid scale is consistent across variables
    v1_n = (v1 - x_min) / x_range
    v2_n = (v2 - y_min) / y_range

    losses = []
    for _ in range(n_cuts):
        for _ in range(max_tries):
            with torch.no_grad():
                cut1 = np.random.beta(cut_beta_alpha, 1.0)
                cut2 = np.random.beta(cut_beta_alpha, 1.0)
            NA, NB, NC, ND = sigmoid_counts(v1_n, v2_n, cut1, cut2, w, scale=scale)
            if (NA.item() > n_events_min and NB.item() > n_events_min and
                    NC.item() > n_events_min and ND.item() > n_events_min):
                break
        else:
            continue  # skip this cut if no valid split found
        num = torch.abs(NA * ND - NB * NC)
        den = (NA * ND + NB * NC + 1e-8) if symmetrize else (NB * NC + 1e-8)
        losses.append(num / den)

    if not losses:
        return torch.tensor(0.0, device=var1.device, dtype=var1.dtype)
    return torch.stack(losses).mean()


def _proxy_md(latent, labels, qcd_label=1):
    """
    Squared Mahalanobis distance with PCA whitening.
    Fitted on QCD events in the current batch (detached).
    Gradients flow through latent only.

    Steps:
      1. Fit μ and Σ on QCD latents.
      2. Eigendecompose Σ = V Λ Vᵀ (torch.linalg.eigh, stable for symmetric matrices).
      3. Build whitening matrix W = V / sqrt(λ): rotates to principal components
         and scales each PC to unit variance.
      4. z = (latent - μ) @ W  — whitened latent, gradients flow through here.
      5. score = ||z||² = squared Euclidean in whitened space = MD² in original space.
    """
    qcd_mask = (labels == qcd_label)
    if qcd_mask.sum() < 2:
        return torch.zeros(latent.size(0), device=latent.device)

    with torch.no_grad():
        bkg_latents = latent[qcd_mask].detach().float()
        mu = bkg_latents.mean(0)
        centered = bkg_latents - mu
        cov = (centered.T @ centered) / bkg_latents.shape[0]
        # eigh returns eigenvalues ascending and eigenvectors as columns
        L, V = torch.linalg.eigh(cov)
        L = L.clamp(min=1e-6)          # guard against near-zero eigenvalues
        W = V / L.sqrt()               # whitening matrix [D, D]

    z = (latent.float() - mu) @ W     # whitened latent [B, D]; grads flow through latent
    scores = (z * z).sum(dim=1)        # ||z||² = MD² in whitened space
    return scores.to(latent.dtype)


def train_epoch(
    encoder, projector, classifier,
    ce_loss_fn, contrastive_loss,
    train_loader, norm_constants, device,
    optimizer, scheduler=None, contrastive_weight=0.05,
    pairwise=False, num_classes=4, scaler=None,
    ae_model=None, ae_reco_weight=1.0, disco_weight=0.0, closure_weight=0.0, qcd_label=1,
):
    """
    One training epoch.

    If ae_model is provided and disco_weight > 0, runs double DisCo:
    - AE forward pass on object-level features  → per-event reco loss (axis 1)
    - Contrastive proxy MD from embeddings       → per-event score    (axis 2)
    - DisCo between the two on QCD events backpropagates through BOTH models.

    Total loss = w_con*contrast + w_ce*ce + ae_reco_weight*ae_reco + disco_weight*disco
    (DataLoader must yield a 4th element: normalised obj features [B, D] when ae_model is set)
    """
    encoder.train(); projector.train(); classifier.train()
    if ae_model is not None:
        ae_model.train()

    total_loss = total_contrast = total_ce = total_ae = total_disco = total_closure = 0.0
    count = 0
    scheduled_contrst_wght  = not (isinstance(contrastive_weight, int) or isinstance(contrastive_weight, float))
    scheduled_disco_wght    = not (isinstance(disco_weight,        int) or isinstance(disco_weight,        float))
    scheduled_closure_wght  = not (isinstance(closure_weight,      int) or isinstance(closure_weight,      float))
    class_metrics = ClassificationMetrics(num_classes)
    mse_no_reduce = torch.nn.MSELoss(reduction="none")

    for batch in train_loader:
        if len(batch) == 4:
            x, mask, labels, obj = batch
            obj = obj.to(device)
        else:
            x, mask, labels = batch
            obj = None

        x = x.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool),
            mask.bool()
        ], dim=1)

        delta_r = delta_r_from_normalized(x, norm_constants) if pairwise else None

        use_amp = (device == "cuda") and (scaler is not None) and scaler.is_enabled()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            # ── contrastive branch ──
            latent = encoder(x, delta_r, mask)
            embeddings = F.normalize(projector(latent), dim=1)

            loss_contrast = contrastive_loss(embeddings, labels)
            logits  = classifier(embeddings)
            loss_ce = ce_loss_fn(logits, labels)

            contrast_weight_value = contrastive_weight.get() if scheduled_contrst_wght else contrastive_weight
            loss = contrast_weight_value * loss_contrast + (1 - contrast_weight_value) * loss_ce

            # ── AE branch ──
            loss_ae = torch.tensor(0.0, device=device)
            ae_reco_per_event = None
            if ae_model is not None and obj is not None:
                recon, _ = ae_model(obj)
                ae_reco_per_event = mse_no_reduce(recon, obj).mean(dim=1)  # [B]
                loss_ae = ae_reco_per_event.mean()
                loss = loss + ae_reco_weight * loss_ae

        # ── DisCo + closure: computed outside autocast for float32 numerical stability ──
        disco_weight_value   = disco_weight.get()   if scheduled_disco_wght   else disco_weight
        closure_weight_value = closure_weight.get() if scheduled_closure_wght else closure_weight

        loss_disco   = torch.tensor(0.0, device=device)
        loss_closure = torch.tensor(0.0, device=device)
        if ae_model is not None and ae_reco_per_event is not None and (disco_weight_value > 0.0 or closure_weight_value > 0.0):
            qcd_mask = (labels == qcd_label)
            if qcd_mask.sum() > 10:
                proxy_md = _proxy_md(latent, labels, qcd_label)
                nw = torch.ones(qcd_mask.sum(), device=device)

                if disco_weight_value > 0.0:
                    loss_disco = distance_corr(
                        ae_reco_per_event[qcd_mask].float(),
                        proxy_md[qcd_mask].float(),
                        nw,
                    )
                    loss = loss + disco_weight_value * loss_disco

                if closure_weight_value > 0.0:
                    loss_closure = closure_loss_batch(
                        ae_reco_per_event[qcd_mask].float(),
                        proxy_md[qcd_mask].float(),
                        nw,
                    )
                    loss = loss + closure_weight_value * loss_closure

        all_params = (
            list(encoder.parameters()) +
            list(projector.parameters()) +
            list(classifier.parameters()) +
            (list(ae_model.parameters()) if ae_model is not None else [])
        )
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if scheduled_contrst_wght:
            contrastive_weight.step()
        if scheduled_disco_wght:
            disco_weight.step()
        if scheduled_closure_wght:
            closure_weight.step()

        bs = x.size(0)
        total_loss     += loss.item() * bs
        total_contrast += loss_contrast.item() * bs
        total_ce       += loss_ce.item() * bs
        total_ae       += loss_ae.item() * bs
        total_disco    += loss_disco.item() * bs
        total_closure  += loss_closure.item() * bs
        count          += bs
        class_metrics.update(logits, labels)

    return {
        "loss":         total_loss     / count,
        "contrast":     total_contrast / count,
        "ce":           total_ce       / count,
        "ae_reco":      total_ae       / count,
        "disco":        total_disco    / count,
        "closure":      total_closure  / count,
        **class_metrics.compute_metrics(),
    }

@torch.no_grad()
def validate_epoch(
    encoder, projector, classifier,
    ce_loss_fn, contrastive_loss,
    val_loader, norm_constants, device,
    contrastive_weight=0.05,
    pairwise=False, num_classes=4,
    ae_model=None, ae_reco_weight=1.0,
):
    """
    Validation pass (no grads). Returns averaged metrics.
    """
    encoder.eval(); projector.eval(); classifier.eval()
    if ae_model is not None:
        ae_model.eval()

    total_loss = total_contrast = total_ce = total_ae = 0.0
    count = 0
    scheduled_contrst_wght = not (isinstance(contrastive_weight, int) or isinstance(contrastive_weight, float))
    class_metrics = ClassificationMetrics(num_classes)
    mse_no_reduce = torch.nn.MSELoss(reduction="none")

    for batch in val_loader:
        if len(batch) == 4:
            x, mask, labels, obj = batch
            obj = obj.to(device)
        else:
            x, mask, labels = batch
            obj = None

        x = x.to(device); mask = mask.to(device); labels = labels.to(device)

        mask = torch.cat([
            torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool),
            mask.bool()
        ], dim=1)

        delta_r = delta_r_from_normalized(x, norm_constants) if pairwise else None

        latent = encoder(x, delta_r, mask)
        embeddings = F.normalize(projector(latent), dim=1)

        loss_contrast = contrastive_loss(embeddings, labels)
        logits  = classifier(embeddings)
        loss_ce = ce_loss_fn(logits, labels)

        contrast_weight_value = contrastive_weight.get() if scheduled_contrst_wght else contrastive_weight
        loss = contrast_weight_value * loss_contrast + (1 - contrast_weight_value) * loss_ce

        loss_ae = torch.tensor(0.0, device=device)
        if ae_model is not None and obj is not None:
            recon, _ = ae_model(obj)
            loss_ae = mse_no_reduce(recon, obj).mean(dim=1).mean()
            loss = loss + ae_reco_weight * loss_ae

        bs = x.size(0)
        total_loss     += loss.item() * bs
        total_contrast += loss_contrast.item() * bs
        total_ce       += loss_ce.item() * bs
        total_ae       += loss_ae.item() * bs
        count          += bs
        class_metrics.update(logits, labels)

    return {
        "loss":    total_loss     / count,
        "contrast": total_contrast / count,
        "ce":      total_ce       / count,
        "ae_reco": total_ae       / count,
        **class_metrics.compute_metrics()
    }

def cosine_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr: float,
        lr_min: float = 0.0,
    ):
    """
    Cosine learning rate schedule with linear warmup.

    - LR starts at lr_min
    - warms up linearly to lr over warmup_steps
    - then decays with cosine back toward lr_min by total_steps
    """
    lr_delta = lr - lr_min
    def lr_lambda(step):
        if step < warmup_steps: # Linearly go from lr_min to lr_max. If no lr_max, then keep lr const at lr_min)
            return (1/lr) * (lr_min + lr_delta * (step / warmup_steps))
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return (1/lr) * (lr - lr_delta * 0.5 * (1 - math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

class linear_warmup_weight:
    """
    Linearly ramps a weight from 0 to target over warmup_steps, then holds at target.
    Step once per optimizer step (i.e., per batch).
    """
    def __init__(self, target: float, warmup_steps: int):
        self.target = target
        self.warmup_steps = warmup_steps
        self._step = 0
        self.current_weight = self._compute(0)

    def _compute(self, step: int) -> float:
        if self.warmup_steps == 0:
            return self.target
        return self.target * min(1.0, step / self.warmup_steps)

    def step(self) -> float:
        self._step += 1
        self.current_weight = self._compute(self._step)
        return self.current_weight

    def get(self) -> float:
        return self.current_weight


class cosine_constrastive_schedule:
    """
    Cosine schedule for the *contrastive weight* (not LR).

    - weight starts at weight_min
    - warms up linearly to weight_max over warmup_steps
    - then decays with cosine back toward weight_min by total_steps

    Stepped once per optimizer step (i.e., per batch).
    """
    def __init__(
        self,
        weight_min: float,
        weight_max: float,
        warmup_steps: int,
        total_steps: int
    ):
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0
        self.current_weight = self._compute(0)

    def _compute(self, step: int) -> float:
        """
        Max: weight_max at end of warmup (step==warmup_steps)
        Min: weight_min at step==total_steps
        0 <= step <= total_steps
        """

        # Clamp [0, total_steps]
        step = max(0, min(step, self.total_steps))

        if step < self.warmup_steps:
            # Linear warmup: min -> max
            t = step / self.warmup_steps
            return self.weight_min + t * (self.weight_max - self.weight_min)

        # Cosine decay: max -> min
        # progress in [0,1] after warmup
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        # cosine from 1 -> -1 maps to 1 -> 0 with 0.5*(1+cos)
        cos_term = 0.5 * (1 + math.cos(math.pi * progress)) # Max: 1, Min: 0
        return self.weight_min + cos_term * (self.weight_max - self.weight_min)

    def step(self) -> float:
        """
        Advance schedule by 1 step and update current_weight.
        Call once per batch (after optimizer.step()).
        """
        self._step += 1
        self.current_weight = self._compute(self._step)
        return self.current_weight

    def get(self) -> float:
        """Return current weight without advancing."""
        return self.current_weight

    def state_dict(self) -> dict:
        return {
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "_step": self._step,
            "current_weight": self.current_weight,
        }

    def load_state_dict(self, state: dict) -> None:
        self.weight_min = float(state["weight_min"])
        self.weight_max = float(state["weight_max"])
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self._step = int(state["_step"])
        self.current_weight = float(state["current_weight"])

class ClassificationMetrics:
    """
    Utility class to track and compute classification metrics (accuracy, precision, recall, F1).
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fn = torch.zeros(self.num_classes, dtype=torch.long)
        self.tn = torch.zeros(self.num_classes, dtype=torch.long)
        self.num_events = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Update counts based on batch predictions and true labels.
        logits: [B, C] raw model outputs, where C = num_classes
        labels: [B] true class indices
        """

        self.num_events += labels.size(0)
        preds = logits.argmax(dim=1)  # [B]

        for cls in range(self.num_classes):
            cls_preds = (preds == cls) # [B] bool tensor where True means class predicted correctly as cls
            cls_labels = (labels == cls) # [B] bool tensor where True means true label is cls
            self.tp[cls] += (cls_preds & cls_labels).sum().item()
            self.fp[cls] += (cls_preds & ~cls_labels).sum().item()
            self.fn[cls] += (~cls_preds & cls_labels).sum().item()
            self.tn[cls] += (~cls_preds & ~cls_labels).sum().item()

    def compute_metrics(self) -> dict:
        # TODO: Add more metrics

        accuracy = (self.tp.sum().item()) / self.num_events if self.num_events > 0 else 0.0

        return {
            "acc": accuracy,
        }
