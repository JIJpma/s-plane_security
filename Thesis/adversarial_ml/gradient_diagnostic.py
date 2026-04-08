"""Quick diagnostic: mean absolute gradient per feature on a batch of attack windows."""

import torch
import pandas as pd
from torch.utils.data import DataLoader

from adversarial_ml.utils import PTPDataset, load_model

MODEL_PATH = "DU_model/Transformer/best_model_prod_only2_tr.3.64.pth"
DATA_PATH = "DU_model/Datasets/Original/prod_successful_announce_attack.csv"
FEATURE_COLS = ["Source", "Destination", "Length", "SequenceID", "MessageType", "Time Interval"]
SLICE_LEN = 64
NHEAD = 3
BATCH_SIZE = 1024


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(MODEL_PATH, device, slice_len=SLICE_LEN, nhead=NHEAD)
    print(f"Loaded model from {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    dataset = PTPDataset(df[FEATURE_COLS], df["Label"], SLICE_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Find a batch that contains attack windows
    inputs, labels = None, None
    for batch_inputs, batch_labels in loader:
        if batch_labels.sum() > 0:
            inputs, labels = batch_inputs, batch_labels
            break

    if inputs is None:
        print("No attack windows found in dataset!")
        return

    inputs = inputs.to(device)
    print(f"Batch shape: {inputs.shape}  |  attack windows: {labels.sum().item()} / {len(labels)}")

    model.eval()
    inputs.requires_grad_(True)

    logits = model(inputs, return_logits=True).squeeze(-1)
    probs = torch.sigmoid(logits)

    print(f"Prob  stats — mean: {probs.mean():.6f}  min: {probs.min():.6f}  max: {probs.max():.6f}")
    saturated_lo = (probs < 0.01).sum().item()
    saturated_hi = (probs > 0.99).sum().item()
    mid_range = ((probs >= 0.01) & (probs <= 0.99)).sum().item()
    print(f"  Saturated near 0: {saturated_lo}  |  near 1: {saturated_hi}  |  mid-range: {mid_range}")
    print(f"Logit stats — mean: {logits.mean():.4f}  min: {logits.min():.4f}  max: {logits.max():.4f}")

    grads = torch.autograd.grad(logits.sum(), inputs, create_graph=False)[0]
    print(f"  grad shape: {grads.shape}  grad abs max: {grads.abs().max():.6f}")

    grad_importance = grads.abs().mean(dim=(0, 1))
    print("\nMean absolute gradient per feature (all windows):")
    for name, val in zip(FEATURE_COLS, grad_importance):
        print(f"  {name:20s}  {val:.2e}")

    # Also show gradient importance for attack-only windows
    attack_mask = labels.bool()
    if attack_mask.any():
        grad_attack = grads[attack_mask].abs().mean(dim=(0, 1))
        print(f"\nMean absolute gradient per feature (attack windows only, n={attack_mask.sum()}):")
        for name, val in zip(FEATURE_COLS, grad_attack):
            print(f"  {name:20s}  {val:.2e}")


if __name__ == "__main__":
    main()
