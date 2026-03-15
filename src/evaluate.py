"""
evaluate.py — Threshold tuning and evaluation metrics

Why threshold tuning matters:
  Default 0.5 threshold is calibrated for balanced data.
  With 3.5% fraud (28:1 ratio), the model needs a lower threshold
  to catch more fraud — even at the cost of more false alarms.

  F2-score (beta=2) weights recall 2x over precision:
  - Missing fraud (FN) is more costly than a false alarm (FP)
  - This matches real business logic in fintech fraud detection

Baseline numbers to beat (from notebooks/04_baseline_model.ipynb):
  avg_precision : 0.6257
  recall_fraud  : 0.6747
  fraud_caught  : 67.47%
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix,
    f1_score, fbeta_score,
)
from typing import Optional


# ------------------------------------------------------------------
# Threshold tuning
# ------------------------------------------------------------------
def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 2.0,
) -> tuple:
    """
    Find threshold that maximises F-beta score.

    beta=2.0 → recall weighted 2x (default — fraud use case)
    beta=1.0 → equal precision/recall
    beta=0.5 → precision weighted (low false alarm use case)

    Returns
    -------
    (best_threshold, metrics_dict)
    """
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, y_proba
    )

    beta2    = beta ** 2
    denom    = beta2 * precisions + recalls + 1e-8
    f_scores = (1 + beta2) * precisions * recalls / denom

    best_idx = int(np.argmax(f_scores))
    best_thr = float(thresholds[best_idx]) \
               if best_idx < len(thresholds) else 0.5

    preds  = (y_proba >= best_thr).astype(int)
    report = classification_report(
        y_true, preds, output_dict=True, zero_division=0
    )

    metrics = {
        'threshold':       round(best_thr, 4),
        f'f{beta}_score':  round(float(f_scores[best_idx]), 4),
        'precision_fraud': round(report.get('1', {}).get('precision', 0), 4),
        'recall_fraud':    round(report.get('1', {}).get('recall', 0), 4),
        'f1_fraud':        round(report.get('1', {}).get('f1-score', 0), 4),
        'auc_roc':         round(roc_auc_score(y_true, y_proba), 4),
        'avg_precision':   round(average_precision_score(y_true, y_proba), 4),
    }

    return best_thr, metrics


# ------------------------------------------------------------------
# Full evaluation report
# ------------------------------------------------------------------
def full_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    save_dir: Optional[str] = None,
    split_name: str = 'val',
) -> dict:
    """
    Full evaluation — metrics + confusion matrix + plots.

    Parameters
    ----------
    y_true     : true labels
    y_proba    : predicted probabilities
    threshold  : classification threshold
    save_dir   : if provided, saves plots here
    split_name : label for plots (train / val / test)

    Returns
    -------
    dict of all metrics
    """
    y_pred = (y_proba >= threshold).astype(int)

    auc  = roc_auc_score(y_true, y_proba)
    ap   = average_precision_score(y_true, y_proba)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    f2   = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'auc_roc':          round(auc, 4),
        'avg_precision':    round(ap, 4),
        'f1_fraud':         round(report.get('1', {}).get('f1-score', 0), 4),
        'f2_fraud':         round(f2, 4),
        'precision_fraud':  round(report.get('1', {}).get('precision', 0), 4),
        'recall_fraud':     round(report.get('1', {}).get('recall', 0), 4),
        'threshold':        round(threshold, 4),
        'true_positives':   int(tp),
        'false_positives':  int(fp),
        'true_negatives':   int(tn),
        'false_negatives':  int(fn),
        'fraud_caught_pct': round(tp / (tp + fn) * 100, 2),
        'false_alarm_rate': round(fp / (fp + tn) * 100, 2),
    }

    # ---------- Console report ----------
    _print_report(metrics, split_name)

    # ---------- Baseline comparison ----------
    _compare_to_baseline(metrics)

    # ---------- Plots ----------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _plot_pr_roc(y_true, y_proba, threshold, save_dir, split_name)
        _plot_score_distribution(y_true, y_proba, threshold, save_dir, split_name)

    return metrics


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _print_report(metrics: dict, split_name: str) -> None:
    print(f"\n{'='*55}")
    print(f"  Evaluation — {split_name}")
    print(f"{'='*55}")
    print(f"  AUC-ROC           : {metrics['auc_roc']:.4f}")
    print(f"  Avg Precision(AP) : {metrics['avg_precision']:.4f}  ← primary metric")
    print(f"  F2 (fraud)        : {metrics['f2_fraud']:.4f}")
    print(f"  Precision (fraud) : {metrics['precision_fraud']:.4f}")
    print(f"  Recall (fraud)    : {metrics['recall_fraud']:.4f}")
    print(f"  Threshold used    : {metrics['threshold']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    TP={metrics['true_positives']:>6,}  "
          f"FN={metrics['false_negatives']:>6,}  "
          f"(fraud caught: {metrics['fraud_caught_pct']}%)")
    print(f"    FP={metrics['false_positives']:>6,}  "
          f"TN={metrics['true_negatives']:>6,}  "
          f"(false alarm rate: {metrics['false_alarm_rate']}%)")
    print(f"{'='*55}\n")


def _compare_to_baseline(metrics: dict) -> None:
    """Compare current metrics against baseline notebook numbers."""
    baseline = {
        'avg_precision':   0.6257,
        'recall_fraud':    0.6747,
        'fraud_caught_pct': 67.47,
        'false_alarm_rate': 3.65,
    }

    print("  vs Baseline (notebooks/04_baseline_model.ipynb):")
    for key, base_val in baseline.items():
        curr_val = metrics.get(key, 0)
        if key == 'false_alarm_rate':
            # Lower is better
            delta = base_val - curr_val
            sign  = '✓' if delta >= 0 else '✗'
        else:
            # Higher is better
            delta = curr_val - base_val
            sign  = '✓' if delta >= 0 else '✗'
        print(f"  {sign} {key:<22}: {curr_val:.4f}  "
              f"(baseline: {base_val:.4f}, Δ{delta:+.4f})")
    print()


def _plot_pr_roc(
    y_true, y_proba, threshold, save_dir, split_name
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Precision-Recall
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[0].plot(rec, prec, color='#E24B4A', lw=2, label=f'AP={ap:.3f}')
    axes[0].axhline(
        y=y_true.mean(), color='gray', linestyle='--',
        label=f'Random ({y_true.mean():.3f})'
    )
    # Mark chosen threshold
    idx = np.argmin(np.abs(thr - threshold))
    axes[0].scatter(
        rec[idx], prec[idx], color='black', s=80, zorder=5,
        label=f'Threshold={threshold:.3f}'
    )
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[1].plot(fpr, tpr, color='#378ADD', lw=2, label=f'AUC={auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'gray', linestyle='--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(fontsize=9)

    plt.suptitle(
        f'Model Performance — {split_name}', fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f'eval_pr_roc_{split_name}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f"Saved → {save_dir}/eval_pr_roc_{split_name}.png")


def _plot_score_distribution(
    y_true, y_proba, threshold, save_dir, split_name
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        y_proba[y_true == 0], bins=80, alpha=0.6,
        color='#378ADD', density=True, label='Legit'
    )
    ax.hist(
        y_proba[y_true == 1], bins=80, alpha=0.6,
        color='#E24B4A', density=True, label='Fraud'
    )
    ax.axvline(
        x=threshold, color='black', linestyle='--', lw=1.5,
        label=f'Threshold = {threshold:.3f}'
    )
    ax.set_xlabel('Fraud probability score')
    ax.set_ylabel('Density')
    ax.set_title(f'Score Distribution by Class — {split_name}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f'eval_score_dist_{split_name}.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f"Saved → {save_dir}/eval_score_dist_{split_name}.png")
