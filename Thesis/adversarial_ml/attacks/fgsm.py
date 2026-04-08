import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DU_model.Transformer.t_size import TransformerNN
from adversarial_ml.attacks.base import AttackResult, BaseAttack


class FGSM(BaseAttack):
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Single-step attack: perturb each feature by ±epsilon * feature_std
    in the direction that maximises the loss w.r.t. the model's own prediction.
    """

    name = "fgsm"

    def _perturb(
        self,
        model: TransformerNN,
        inputs: torch.Tensor,
        clean_preds: torch.Tensor,
        epsilon: float,
        feat_std: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs.clone().detach().requires_grad_(True)
        logits = model(x, return_logits=True).squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, clean_preds.float())
        loss.backward()
        perturbation = epsilon * feat_std * x.grad.sign()
        return (x + perturbation).detach()

    def evaluate(
        self,
        model: TransformerNN,
        loader: DataLoader,
        device: torch.device,
        epsilon: float,
        feat_std: torch.Tensor,
    ) -> AttackResult:
        all_clean, all_adv, all_labels = [], [], []

        model.eval()
        for inputs, labels in loader:
            inputs = inputs.to(device)

            with torch.no_grad():
                clean_probs = model(inputs).squeeze(-1)
            clean_preds = (clean_probs >= 0.5).float()

            x_adv = self._perturb(model, inputs, clean_preds, epsilon, feat_std)

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
