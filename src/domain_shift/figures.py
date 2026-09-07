"""
Phase 14 figures.

Limited to scientifically useful plots. The SAME frozen plotting functions are
applied to internal and external data, so no styling or threshold is chosen
after external inspection. matplotlib is used in Agg mode with fixed styling.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np


def _per_bin_mae(model_list: List[str], domain: Dict, n_bins: int) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for m in model_list:
        out[m] = [
            float(np.mean(np.abs(
                domain["y_true"][domain["activity_bin"] == b]
                - domain["pred"][m][domain["activity_bin"] == b]
            ))) if np.sum(domain["activity_bin"] == b) > 0 else np.nan
            for b in range(n_bins)
        ]
    return out


def fig1_gc_distribution(internal, external, gc_edges, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(internal["gc30"], bins=30, density=True, alpha=0.6, label="internal validation")
    ax.hist(external["gc30"], bins=30, density=True, alpha=0.6, label="external test")
    for e in gc_edges:
        ax.axvline(e, color="k", ls="--", lw=0.8)
    ax.set_xlabel("GC content (30-mer)")
    ax.set_ylabel("Density")
    ax.set_title("GC distribution: internal vs external (frozen bin edges)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fig2_activity_distribution(internal, external, activity_edges, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = activity_edges[1:-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(internal["y_true"], bins=30, density=True, alpha=0.6, label="internal validation")
    ax.hist(external["y_true"], bins=30, density=True, alpha=0.6, label="external test")
    for e in split:
        ax.axvline(e, color="k", ls="--", lw=0.8)
    ax.set_xlabel("True activity")
    ax.set_ylabel("Density")
    ax.set_title("Activity distribution: internal vs external (fixed bins)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fig3_mae_by_gc_bin(internal, external, gc_edges, model, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bins = len(gc_edges) + 1
    bins = np.arange(n_bins)
    int_mae = np.array([np.mean(np.abs(internal["y_true"][internal["gc_bin"] == b]
                                       - internal["pred"][model][internal["gc_bin"] == b]))
                        if np.sum(internal["gc_bin"] == b) else np.nan for b in bins])
    ext_mae = np.array([np.mean(np.abs(external["y_true"][external["gc_bin"] == b]
                                       - external["pred"][model][external["gc_bin"] == b]))
                        if np.sum(external["gc_bin"] == b) else np.nan for b in bins])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(n_bins)
    ax.bar(x - 0.2, int_mae, 0.4, label="internal validation", color="C0", alpha=0.9)
    ax.bar(x + 0.2, ext_mae, 0.4, label="external test", color="C2", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in range(1, n_bins + 1)])
    ax.set_xlabel("GC bin (1=lowest GC, 5=highest, frozen edges)")
    ax.set_ylabel("MAE")
    ax.set_title(f"MAE across frozen GC bins ({model})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fig4_mae_by_activity_bin(internal, external, model, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    int_mae = _per_bin_mae([model], internal, 5)[model]
    ext_mae = _per_bin_mae([model], external, 5)[model]
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, int_mae, 0.4, label="internal validation", color="C0", alpha=0.9)
    ax.bar(x + 0.2, ext_mae, 0.4, label="external test", color="C2", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"])
    ax.set_xlabel("True-activity bin (fixed)")
    ax.set_ylabel("MAE")
    ax.set_title(f"MAE across frozen activity bins ({model})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fig5_pred_vs_true(internal, external, models, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(13, 8), sharex=True, sharey=True)
    domains = [("internal validation", internal, 0), ("external test", external, 1)]
    for model, ax in zip(models, axes[0]):
        ax.scatter(internal["y_true"], internal["pred"][model], s=8, alpha=0.4, c="C0")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{model} (internal)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for model, ax in zip(models, axes[1]):
        ax.scatter(external["y_true"], external["pred"][model], s=8, alpha=0.4, c="C2")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{model} (external)")
    for ax in axes[0]:
        ax.set_xlabel("True activity")
        ax.set_ylabel("Predicted")
    for ax in axes[1]:
        ax.set_xlabel("True activity")
        ax.set_ylabel("Predicted")
    fig.suptitle("Prediction vs true activity (frozen activity regions)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fig6_model_comparison(internal, external, models, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    int_tab = _per_bin_mae(models, internal, 5)
    ext_tab = _per_bin_mae(models, external, 5)
    labels = ["[0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    n = len(models)
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / (2 * n)
    for i, (m, c) in enumerate(zip(models, ["C0", "C1", "C3"])):
        ax.bar(x - 0.4 + i * width, int_tab[m], width, label=f"{m} (int)", color=c, alpha=0.6)
        ax.bar(x + 0.4 - (n - i) * width, ext_tab[m], width, label=f"{m} (ext)", color=c, alpha=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("True-activity bin")
    ax.set_ylabel("MAE")
    ax.set_title("RF/XGB/CNN MAE across domain slices")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fig7_composition_summary(kmer_js, null_band, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kvals = sorted(int(k) for k in kmer_js.keys())
    js_vals = [kmer_js[str(k)]["js_divergence"] for k in kvals]
    null_lo = [null_band[str(k)]["null_band_2.5pct"] for k in kvals]
    null_hi = [null_band[str(k)]["null_band_97.5pct"] for k in kvals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(kvals))
    ax.bar(x - 0.2, js_vals, 0.4, label="external vs internal-train JS", color="C2")
    ax.bar(x + 0.2, null_lo, 0.4, label="internal null 2.5pct", color="C1", alpha=0.7)
    ax.errorbar(x + 0.2, null_lo, yerr=[np.zeros(len(null_lo)), np.array(null_hi) - np.array(null_lo)],
                fmt="none", ecolor="C1", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}-mer" for k in kvals])
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_title("Sequence-composition divergence summary (exploratory)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def fig8_joint_heatmap(cells, n_gc, n_act, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = np.full((n_gc, n_act), np.nan)
    for c in cells:
        mat[c["gc_bin"], c["activity_bin"]] = c["delta_mae"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_xlabel("Activity bin")
    ax.set_ylabel("GC bin (1=lowest, 5=highest)")
    ax.set_title("Delta MAE (ext - int) per common-support cell")
    fig.colorbar(im, ax=ax, label="Delta MAE")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)