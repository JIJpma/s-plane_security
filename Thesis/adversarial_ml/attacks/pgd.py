"""Projected Gradient Descent with PTP protocol constraints.

Ensures adversarial perturbations remain valid attack traffic by:
- Only perturbing features the attacker actually controls (inter-arrival time,
  optionally SequenceID)
- Only modifying packets the attacker sends (malicious packets), not legitimate
  master traffic interleaved in the sequence
- Enforcing physical constraints (non-negative timing, integer SeqIDs)
- Preserving protocol-fixed fields (MACs, MessageType, Length)

Reference: Madry et al., "Towards Deep Learning Models Resistant to
Adversarial Attacks," ICLR 2018.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Optional

from DU_model.Transformer.t_size import TransformerNN
from adversarial_ml.attacks.base import AttackResult, BaseAttack


# ---------------------------------------------------------------------------
# Feature column indices (must match the CSV / dataset ordering)
# ---------------------------------------------------------------------------
IDX_SRC = 0   # Source MAC (integer-encoded)
IDX_DST = 1   # Destination MAC (integer-encoded)
IDX_LEN = 2   # Ethernet frame length
IDX_SEQ = 3   # PTP SequenceID
IDX_MSG = 4   # PTP MessageType
IDX_IAT = 5   # Inter-arrival time

NUM_FEATURES = 6


@dataclass
class PerturbConfig:
    """Controls which features and which packets are eligible for perturbation.

    Attributes:
        perturb_iat:       Allow perturbation of inter-arrival time.
        perturb_seq_id:    Allow perturbation of SequenceID.
        attacker_src_id:   Integer-encoded MAC of the attacker.  When set, only
                           packets whose Source matches this value are perturbed.
                           When ``None``, falls back to ``use_labels_as_mask``.
        use_labels_as_mask:  If True (and ``attacker_src_id`` is None), use the
                           per-packet ground-truth label to decide which packets
                           the attacker "owns" (label == 1 → attacker packet).
        iat_min:           Physical lower bound for inter-arrival time.
        iat_max:           Optional upper bound (e.g., max observed in training).
        seq_id_jitter:     Maximum integer deviation allowed for SequenceID.
    """
    perturb_iat: bool = True
    perturb_seq_id: bool = False
    attacker_src_id: Optional[int] = None
    use_labels_as_mask: bool = True
    iat_min: float = 0.0
    iat_max: Optional[float] = None
    seq_id_jitter: int = 5


class PGD(BaseAttack):
    """Projected Gradient Descent with PTP-protocol-aware constraints."""

    name = "pgd"

    def __init__(
        self,
        steps: int = 10,
        alpha_ratio: float = 0.25,
        random_start: bool = True,
        config: Optional[PerturbConfig] = None,
    ):
        """
        Args:
            steps:        Number of iterative gradient steps.
            alpha_ratio:  Step size as a fraction of epsilon (alpha = eps * alpha_ratio).
            random_start: Initialise from a random point inside the eps-ball.
            config:       Protocol-aware perturbation constraints.
        """
        self.steps = steps
        self.alpha_ratio = alpha_ratio
        self.random_start = random_start
        self.cfg = config or PerturbConfig()

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def _feature_mask(self, device: torch.device) -> torch.Tensor:
        """Boolean mask over the feature dimension (shape [NUM_FEATURES]).

        True = feature is eligible for perturbation.
        """
        mask = torch.zeros(NUM_FEATURES, dtype=torch.bool, device=device)
        if self.cfg.perturb_iat:
            mask[IDX_IAT] = True
        if self.cfg.perturb_seq_id:
            mask[IDX_SEQ] = True
        return mask

    def _packet_mask(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Boolean mask over the sequence dimension (shape [batch, seq_len]).

        True = packet is owned by the attacker and may be perturbed.
        """
        batch, seq_len, _ = inputs.shape
        device = inputs.device

        if self.cfg.attacker_src_id is not None:
            # Identify attacker packets by source MAC
            return inputs[..., IDX_SRC] == self.cfg.attacker_src_id
        elif self.cfg.use_labels_as_mask and labels is not None:
            if labels.dim() == 1:
                # Per-sequence label: expand to all packets in the sequence
                return labels.bool().unsqueeze(1).expand(batch, seq_len)
            else:
                # Per-packet labels already [batch, seq_len]
                return labels.bool()
        else:
            # Fallback: allow perturbation on every packet
            return torch.ones(batch, seq_len, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def _project(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
        scaled_eps: torch.Tensor,
        feat_mask: torch.Tensor,
        pkt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project onto the intersection of the L-inf ball and the valid domain.

        Steps:
        1. Clamp into the L-inf epsilon-ball around x_orig.
        2. Enforce physical constraints (non-negative IAT, integer SeqID).
        3. Restore all frozen features / frozen packets to their originals.
        """
        # --- L-inf projection ---
        x_adv = torch.max(torch.min(x_adv, x_orig + scaled_eps), x_orig - scaled_eps)

        # --- Domain constraints on inter-arrival time ---
        x_adv[..., IDX_IAT] = x_adv[..., IDX_IAT].clamp(min=self.cfg.iat_min)
        if self.cfg.iat_max is not None:
            x_adv[..., IDX_IAT] = x_adv[..., IDX_IAT].clamp(max=self.cfg.iat_max)

        # --- SequenceID must stay integer-valued and within jitter range ---
        if self.cfg.perturb_seq_id:
            seq_orig = x_orig[..., IDX_SEQ]
            seq_adv = x_adv[..., IDX_SEQ].round()
            seq_adv = seq_adv.clamp(
                min=(seq_orig - self.cfg.seq_id_jitter),
                max=(seq_orig + self.cfg.seq_id_jitter),
            )
            # SeqID must not go negative
            seq_adv = seq_adv.clamp(min=0)
            x_adv[..., IDX_SEQ] = seq_adv

        # --- Restore frozen features (protocol-fixed columns) ---
        frozen_feat = ~feat_mask                           # [NUM_FEATURES]
        x_adv[..., frozen_feat] = x_orig[..., frozen_feat]

        # --- Restore frozen packets (not owned by attacker) ---
        frozen_pkt = ~pkt_mask                             # [batch, seq_len]
        frozen_pkt_expanded = frozen_pkt.unsqueeze(-1).expand_as(x_adv)
        x_adv[frozen_pkt_expanded] = x_orig[frozen_pkt_expanded]

        return x_adv

    # ------------------------------------------------------------------
    # Core PGD loop
    # ------------------------------------------------------------------
    def _perturb(
        self,
        model: TransformerNN,
        inputs: torch.Tensor,
        clean_preds: torch.Tensor,
        epsilon: float,
        feat_std: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run PGD to find adversarial examples within protocol constraints.

        Args:
            model:       Target classifier.
            inputs:      Clean input tensor [batch, seq_len, NUM_FEATURES].
            clean_preds: Model predictions on clean inputs (used as target to flip).
            epsilon:     Perturbation budget in units of feature standard deviations.
            feat_std:    Per-feature standard deviation [NUM_FEATURES].
            labels:      Optional per-packet or per-sequence ground truth labels.

        Returns:
            Adversarial inputs with the same shape as ``inputs``.
        """
        device = inputs.device
        scaled_eps = epsilon * feat_std.to(device)    # [NUM_FEATURES]
        alpha = self.alpha_ratio * scaled_eps          # step size

        feat_mask = self._feature_mask(device)         # [NUM_FEATURES]
        pkt_mask = self._packet_mask(inputs, labels)   # [batch, seq_len]

        x_adv = inputs.clone().detach()

        # --- Optional random start inside the eps-ball ---
        if self.random_start:
            noise = torch.empty_like(x_adv).uniform_(-1, 1) * scaled_eps
            x_adv = x_adv + noise
            x_adv = self._project(x_adv, inputs, scaled_eps, feat_mask, pkt_mask)

        # --- Iterative steps (attack logits to avoid sigmoid gradient masking) ---
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv, return_logits=True).squeeze(-1)
            loss = nn.BCEWithLogitsLoss()(logits, clean_preds.float())
            loss.backward()

            grad_sign = x_adv.grad.sign()

            # Zero gradient for features we cannot change
            grad_sign[..., ~feat_mask] = 0.0

            # Zero gradient for packets the attacker does not own
            frozen_pkt = ~pkt_mask
            grad_sign[frozen_pkt.unsqueeze(-1).expand_as(grad_sign)] = 0.0

            x_adv = x_adv.detach() + alpha * grad_sign
            x_adv = self._project(x_adv, inputs, scaled_eps, feat_mask, pkt_mask)
            x_adv = x_adv.detach()

        return x_adv

    # ------------------------------------------------------------------
    # Public evaluation entry point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        model: TransformerNN,
        loader: DataLoader,
        device: torch.device,
        epsilon: float,
        feat_std: torch.Tensor,
    ) -> AttackResult:
        """Run PGD on every batch and collect clean vs adversarial predictions.

        The loader is expected to yield ``(inputs, labels)`` tuples where
        ``inputs`` has shape [batch, seq_len, NUM_FEATURES].
        """
        all_clean, all_adv, all_labels = [], [], []

        model.eval()
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels_dev = labels.to(device)

            with torch.no_grad():
                clean_probs = model(inputs).squeeze(-1)
            clean_preds = (clean_probs >= 0.5).float()

            x_adv = self._perturb(
                model, inputs, clean_preds, epsilon, feat_std, labels=labels_dev,
            )

            with torch.no_grad():
                adv_probs = model(x_adv).squeeze(-1)
            adv_preds = (adv_probs >= 0.5).long()

            all_clean.append(clean_preds.cpu().long())
            all_adv.append(adv_preds.cpu())
            all_labels.append(labels)

        return AttackResult(
            epsilon=epsilon,
            clean_preds=torch.cat(all_clean),
            adv_preds=torch.cat(all_adv),
            ground_truth=torch.cat(all_labels),
        )