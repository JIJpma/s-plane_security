"""
Adversarial ML evaluation harness.

Usage examples:
    python -m adversarial_ml.main --attack fgsm
    python -m adversarial_ml.main --attack pgd --pgd-steps 20 --pgd-alpha 0.1
    python -m adversarial_ml.main --attack fgsm --epsilons 0.01 0.1 0.5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from adversarial_ml.attacks import get_attack, ATTACK_REGISTRY
from adversarial_ml.utils import PTPDataset, load_model, predict

DEFAULTS = {
    "model_path": "DU_model/Transformer/best_model_prod_only2_tr.3.64.pth",
    "data_path": "DU_model/Datasets/Original/prod_successful_announce_attack.csv",
    "feature_cols": ["Source", "Destination", "Length", "SequenceID", "MessageType", "Time Interval"],
    "slice_len": 64,
    "nhead": 3,
    "batch_size": 1024,
    "epsilons": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
}

TARGET_NAMES = ["benign", "attack"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial robustness evaluation")
    p.add_argument("--attack", required=True, choices=list(ATTACK_REGISTRY), help="Attack method")
    p.add_argument("--model-path", default=DEFAULTS["model_path"])
    p.add_argument("--data-path", default=DEFAULTS["data_path"])
    p.add_argument("--slice-len", type=int, default=DEFAULTS["slice_len"])
    p.add_argument("--nhead", type=int, default=DEFAULTS["nhead"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epsilons", type=float, nargs="+", default=DEFAULTS["epsilons"])
    # PGD-specific
    p.add_argument("--pgd-steps", type=int, default=10, help="PGD iteration count")
    p.add_argument("--pgd-alpha", type=float, default=0.25, help="PGD step-size ratio (alpha = eps * ratio)")
    p.add_argument("--pgd-no-random-start", action="store_true", help="Disable random initialisation for PGD")
    return p.parse_args(argv)


def save_results(run_dir: Path, config: dict, metrics: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nResults saved to {run_dir}")


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model_path, device, slice_len=args.slice_len, nhead=args.nhead)
    print(f"Loaded model from {args.model_path}")

    df = pd.read_csv(args.data_path)
    feature_cols = DEFAULTS["feature_cols"]
    labels = df["Label"]
    features = df[feature_cols]
    print(f"Dataset: {len(df)} rows, {labels.sum()} labeled as attack")

    dataset = PTPDataset(features, labels, args.slice_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- baseline evaluation ---
    probs, targets = predict(model, loader, device)
    preds = (probs >= 0.5).long()
    print(f"\nBaseline — flagged {preds.sum().item()} / {len(preds)} windows as attack")
    print(f"Mean confidence: {probs.mean():.4f}  |  Max: {probs.max():.4f}  |  Min: {probs.min():.4f}\n")
    print(classification_report(targets, preds, target_names=TARGET_NAMES, zero_division=0))

    # --- prepare output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent / "results" / f"{timestamp}_{args.attack}"

    # --- instantiate attack ---
    attack_kwargs = {}
    if args.attack == "pgd":
        attack_kwargs = dict(
            steps=args.pgd_steps,
            alpha_ratio=args.pgd_alpha,
            random_start=not args.pgd_no_random_start,
        )
    attack = get_attack(args.attack, **attack_kwargs)

    feat_std = torch.tensor(features.std().values, dtype=torch.float32, device=device).clamp(min=1e-8)
    print(f"Feature std: {dict(zip(feature_cols, feat_std.cpu().tolist()))}")

    # --- epsilon sweep ---
    evasion_rates = []
    sweep_data = []

    header = f"{attack.name.upper()} Adversarial Robustness Evaluation  (eps in units of σ)"
    print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")

    last_result = None
    for eps in args.epsilons:
        result = attack.evaluate(model, loader, device, eps, feat_std)
        evasion_rates.append(result.evasion_rate)
        sweep_data.append({"epsilon": eps, "flipped": result.flipped, "total": result.total, "evasion_rate": result.evasion_rate})
        print(f"  eps={eps:<5.2f}σ |  flipped {result.flipped:>6d} / {result.total}  |  evasion rate {result.evasion_rate:.4f}")
        last_result = result

    # --- classification report at largest epsilon ---
    print(f"\nClassification report after {attack.name.upper()} (eps={args.epsilons[-1]}):")
    report_dict = classification_report(
        last_result.ground_truth, last_result.adv_preds,
        target_names=TARGET_NAMES, zero_division=0, output_dict=True,
    )
    print(classification_report(
        last_result.ground_truth, last_result.adv_preds,
        target_names=TARGET_NAMES, zero_division=0,
    ))

    # --- confusion matrix ---
    run_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(last_result.ground_truth, last_result.adv_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES)
    disp.plot(cmap="Blues")
    plt.title(f"After {attack.name.upper()} (eps={args.epsilons[-1]}σ)")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # --- evasion-rate plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(args.epsilons, evasion_rates, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Epsilon (σ)")
    ax.set_ylabel("Evasion Rate (fraction of flipped predictions)")
    ax.set_title(f"{attack.name.upper()} Evasion Rate vs Perturbation Budget")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "evasion_rate.png", dpi=150)
    plt.close()

    # --- persist config + metrics ---
    config = { 
        "attack": args.attack,
        "model_path": args.model_path,
        "data_path": args.data_path,
        "feature_cols": feature_cols,
        "slice_len": args.slice_len,
        "nhead": args.nhead,
        "batch_size": args.batch_size,
        "epsilons": args.epsilons,
        "device": str(device),
        "timestamp": timestamp,
    }
    if args.attack == "pgd":
        config["pgd_steps"] = args.pgd_steps
        config["pgd_alpha_ratio"] = args.pgd_alpha
        config["pgd_random_start"] = not args.pgd_no_random_start

    metrics = {
        "baseline": {
            "flagged": int(preds.sum().item()),
            "total": int(len(preds)),
            "mean_confidence": float(probs.mean()),
        },
        "sweep": sweep_data,
        "classification_report_last_eps": report_dict,
    }
    save_results(run_dir, config, metrics)


if __name__ == "__main__":
    main()
