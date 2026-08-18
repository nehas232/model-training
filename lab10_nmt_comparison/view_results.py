"""
Lab 10 (A1): View final results summary
==========================================
Prints BLEU scores and sample translations, and opens all 4 loss curve
images so you can review everything produced by the project in one go.

Usage:
    python view_results.py
"""

import csv
import os
import webbrowser

RESULTS_DIR = "results"


def print_bleu_scores():
    path = os.path.join(RESULTS_DIR, "bleu_scores.csv")
    print("=" * 60)
    print("BLEU SCORES")
    print("=" * 60)
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in sorted(rows, key=lambda r: float(r['bleu_score']), reverse=True):
            print(f"  {row['model']:<20} {float(row['bleu_score']):.4f}")
    print()


def print_sample_translations(n=5):
    path = os.path.join(RESULTS_DIR, "sample_translations.csv")
    print("=" * 60)
    print(f"SAMPLE TRANSLATIONS (showing {n})")
    print("=" * 60)
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            print(f"\n--- Example {i+1} ---")
            print(f"  Source:    {row['source']}")
            print(f"  Reference: {row['reference']}")
            for model_key in ['RNN', 'EncDec', 'Additive', 'Multiplicative']:
                if model_key in row:
                    print(f"  {model_key:>15}: {row[model_key]}")
    print()


def open_loss_curves():
    print("=" * 60)
    print("OPENING LOSS CURVE IMAGES")
    print("=" * 60)
    curve_files = [
        "loss_curve_rnn.png",
        "loss_curve_encdec.png",
        "loss_curve_additive.png",
        "loss_curve_multiplicative.png",
    ]
    for fname in curve_files:
        path = os.path.abspath(os.path.join(RESULTS_DIR, fname))
        if os.path.exists(path):
            print(f"  Opening {fname} ...")
            webbrowser.open(f"file:///{path}")
        else:
            print(f"  [missing] {fname}")
    print()


def main():
    print_bleu_scores()
    print_sample_translations(n=5)
    open_loss_curves()
    print("Done. Full report: report.docx")


if __name__ == '__main__':
    main()