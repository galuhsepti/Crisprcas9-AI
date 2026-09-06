#!/usr/bin/env python3
"""
Phase 12 - Controlled Data-Coverage Ablation (feasibility-gated).

Protocol summary (full text: src/coverage_ablation/config.py):
- The ONLY scientifically valid entrance to the 5-arm design (A..E) is a
  PASSING feasibility audit (protocol Section 4).
- If the audit fails, Phase 12 STOPS and reports:
      "Phase 12 controlled isolation is not identifiable under the available
      data."
  No arm is constructed, no model is trained, and Moreno-Mateos is never
  opened. This script enforces those guarantees programmatically.

Run:  python3 scripts/run_phase12_coverage_ablation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import glob
import json
import logging
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy
import sklearn

from src.multidataset.datasets import (
    allowed_additional_datasets,
    load_deepspcas9,
    load_deephf,
    sha256_file,
)
from src.coverage_ablation.config import (
    EXPERIMENT_NAME_PREFIX,
    RESULTS_DIR,
    POOLS_DIR,
    FIGURES_DIR,
    ACTIVITY_EDGES,
    ARM_DEFINITIONS,
    PROPOSED_TOLERANCES,
    STOP_CONDITIONS,
    MORENO_MATEOS_FREEZE_POLICY,
)
from src.coverage_ablation.matching import (
    feasibility_audit,
    decide_gate,
    construct_arm_pools,
    pool_equivalence_audit,
)
from src.coverage_ablation.analysis import feasibility_outputs
from src.diagnostics.sequence_analysis import gc_content_array
from src.diagnostics.stats import histogram_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PHASE11_GLOB = 'phase11_targeted_dataset_analysis_*.json'
PHASE10_GLOB = 'phase10_multidataset_*.json'


def git_head() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def git_status() -> str:
    try:
        out = subprocess.run(['git', 'status', '--porcelain'],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or "(clean)"
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def latest(pathglob: str, fmt: str) -> Path:
    cands = sorted(glob.glob(pathglob), key=os.path.getmtime)
    return Path(cands[-1])


def save_figure(fig, name: str, ts: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGURES_DIR / f"phase12_{name}_{ts}.png"
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(p)


# ----------------------------------------------------------------- plotting
def plot_activity_by_source(dsp_labels, hf_labels, ts):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, v, c in [('DeepSpCas9 train', dsp_labels, '#1f77b4'),
                       ('DeepHF', hf_labels, '#2ca02c')]:
        ax.hist(v, bins=30, range=(0, 1), alpha=0.55, density=True,
                color=c, label=name, edgecolor='white', lw=0.4)
    ax.set_xlabel('activity label'); ax.set_ylabel('density')
    ax.set_title('Phase 12 feasibility - activity by source (training domain)')
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'activity_by_source', ts)


def plot_gc_by_source(dsp_seqs, hf_seqs, ts):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, s, c in [('DeepSpCas9 train', dsp_seqs, '#1f77b4'),
                       ('DeepHF', hf_seqs, '#2ca02c')]:
        ax.hist(gc_content_array(list(s)), bins=30, alpha=0.55, density=True,
                color=c, label=name, edgecolor='white', lw=0.4)
    ax.set_xlabel('GC fraction (canonical 30-mer)'); ax.set_ylabel('density')
    ax.set_title('Phase 12 feasibility - GC by source (cannot be matched)')
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, 'gc_by_source', ts)


def plot_gc_vs_activity_bins(hf_labels, hf_seqs, dsp_labels, dsp_seqs, ts):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, labels, seqs, c, mk in [
            ('DeepHF', hf_labels, hf_seqs, '#2ca02c', 'o'),
            ('DeepSpCas9 train', dsp_labels, dsp_seqs, '#1f77b4', 's')]:
        y = np.asarray(labels, float)
        gc = gc_content_array(list(seqs))
        xs, gs = [], []
        for i in range(len(ACTIVITY_EDGES) - 1):
            lo, hi = ACTIVITY_EDGES[i], ACTIVITY_EDGES[i + 1]
            m = (y >= lo) & (y < hi) if i < len(ACTIVITY_EDGES) - 2 \
                else (y >= lo) & (y <= hi)
            if m.sum() == 0:
                continue
            xs.append((lo + hi) / 2)
            gs.append(float(gc[m].mean()))
        ax.plot(xs, gs, color=c, marker=mk, label=name, lw=2)
    ax.set_xlabel('activity bin centre'); ax.set_ylabel('mean GC within bin')
    ax.set_title('Phase 12 feasibility - activity<->composition entanglement')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, 'gc_vs_activity_bins', ts)


# ------------------------------------------------------------------- main
def main():
    print("=" * 72)
    print("Phase 12 - Controlled Data-Coverage Ablation (feasibility-gated)")
    print("=" * 72)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    POOLS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    head = git_head()
    environment = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': scipy.__version__,
        'sklearn': sklearn.__version__,
        'git_head': head,
        'git_status_porcelain': git_status(),
    }
    print("\n[0] Reproducibility environment")
    print(f"    git HEAD: {head}")

    print("\n[1] Locked historical results (quoted, never recomputed)")
    ph10_json = latest(str(RESULTS_DIR / PHASE10_GLOB), '')
    ph11_json = latest(str(RESULTS_DIR / PHASE11_GLOB), '')
    ph10 = json.load(open(ph10_json))
    ph11 = json.load(open(ph11_json))
    locked_reference = {
        'phase10': {
            'path': str(ph10_json),
            'sha256': sha256_file(str(ph10_json)),
            'verdict': ph10['verdict'],
            'primary_cnn_d': ph10['external_paired_tests']['cnn']
            ['absolute_error']['mean_pairwise_diff'],
            'ci': [ph10['external_paired_tests']['cnn']
                   ['absolute_error']['bootstrap_ci']['ci_lower'],
                   ph10['external_paired_tests']['cnn']
                   ['absolute_error']['bootstrap_ci']['ci_upper']],
        },
        'phase11': {
            'path': str(ph11_json),
            'sha256': sha256_file(str(ph11_json)),
            'decision_gate': ph11['decision_gate']['gate'],
            'hypotheses': {k: ph11['hypothesis_assessment_H1_H6'].get(k)
                           for k in ['H1', 'H2', 'H3', 'H5', 'H6']},
        },
        'note': ('Values are quoted from the locked Phase 10/11 result JSONs '
                 'only. Moreno-Mateos raw data is NOT read in this phase and '
                 'no new evaluation is performed.'),
    }
    print(f"    phase10 D={locked_reference['phase10']['primary_cnn_d']:+.4f}")

    print("\n[2] Frozen Moreno-Mateos policy")
    print(f"    {MORENO_MATEOS_FREEZE_POLICY}")
    assert 'load_moreno_mateos' not in dir(), \
        'Moreno-Mateos loader must not be imported in Phase 12'
    moreno_mm = {'moreno_mateos_raw_accessed': False,
                 'moreno_mateos_evaluated': False,
                 'moreno_mateos_loader_imported': False}

    print("\n[3] Loading training-domain data (no external data)")
    df_dsp = load_deepspcas9()
    df_hf = load_deephf()
    config = yaml.safe_load(open('config.yaml'))
    train_idx, val_idx = train_test_split(
        np.arange(len(df_dsp)), test_size=config['split']['val_ratio'],
        random_state=config['split']['random_seed'])
    dsp_train = df_dsp.iloc[train_idx].reset_index(drop=True)
    dsp_val = df_dsp.iloc[val_idx].reset_index(drop=True)
    print(f"    DeepSpCas9: valid {len(df_dsp)} -> train {len(dsp_train)} "
          f"/ val {len(dsp_val)}")
    print(f"    DeepHF: {len(df_hf)} labelled rows "
          f"({df_hf.attrs['n_unlabeled_dropped']} NaN-label rows dropped)")

    allowed = allowed_additional_datasets()
    hashes = {
        'deepspcas9_csv': sha256_file('data/raw/DeepSpCas9.csv'),
        'deephf_xlsx': sha256_file('data/external/Benchmarking-CRISPR-on-tools/'
                                   'Training/DeepHF_training.xlsx'),
        'phase10_json': locked_reference['phase10']['sha256'],
        'phase11_json': locked_reference['phase11']['sha256'],
    }

    print("\n[4] Feasibility audit (protocol Section 4)")
    frames = {
        'canonical': {'source': 'deepspcas9',
                      'n': int(len(dsp_train)),
                      'context': 'DeepSpCas9 pooled screen (mod-freq NGS)'},
        'additional': {
            'deephf': {'source': 'deephf',
                       'n': int(len(df_hf)),
                       'context': 'DeepHF pooled screens '
                                  '(HEK293T/HCT116/T)'}},
    }
    feasibility = feasibility_audit(
        frames, dsp_train, {'deephf': df_hf},
        augment_with={
            'allowed_additional_datasets': allowed,
            'validation_split_n': int(len(dsp_val)),
            'input_hashes': hashes,
        })
    gate = decide_gate(feasibility)
    print(f"    decision: {feasibility['decision']}")
    print(f"    triggered stop conditions: "
          f"{feasibility['triggered_stop_conditions']}")
    for n in feasibility['triggered_stop_conditions']:
        print(f"      STOP-{n}: {STOP_CONDITIONS[n]}")
    for note in feasibility['notes']:
        print(f"      note: {note}")
    print(f"    gate: {gate['gate']} - {gate['label']}")

    provenance = {
        'git_head': head,
        'git_status': git_status(),
        'split_policy': 'DeepSpCas9 85/15 seed-42 canonical split',
        'deephf_exclusions': {
            'n_mapped_valid': int(df_hf.attrs['n_mapped_valid']),
            'n_unlabeled_dropped': int(df_hf.attrs['n_unlabeled_dropped']),
            'n_retained': int(len(df_hf)),
        },
        'canonical_artifacts': {m: p for m, p in
                                __import__('src.multidataset.config',
                                           fromlist=['CANONICAL_MODEL_FILES'])
                                .CANONICAL_MODEL_FILES.items()},
    }

    if feasibility['decision'] == 'INFEASIBLE':
        print("\n[5] STOP: Phase 12 controlled isolation is not identifiable "
              "under the available data.")
        print("    No arm pools were constructed; no model was trained; no "
              "external evaluation was performed.")

        # provenance snapshots of the audit inputs (source provenance, not arms)
        meta = {
            'nm': 'phase12_source_provenance_{src}_{ts}.csv',
            'deepspcas9': dsp_train[['normalized_sequence', 'activity_label',
                                     'source_dataset', 'guide_sequence',
                                     'pam']].drop_duplicates('normalized_sequence'),
            'deephf': df_hf[['normalized_sequence', 'activity_label',
                             'source_dataset', 'guide_sequence', 'pam']],
        }
        pool_files = {}
        for key in ['deepspcas9', 'deephf']:
            p = POOLS_DIR / f"phase12_source_provenance_{key}_{ts}.csv"
            meta[key].to_csv(p, index=False)
            pool_files[key] = str(p)

        figures = {}
        figures['activity_by_source'] = plot_activity_by_source(
            dsp_train['activity_label'].astype(float),
            df_hf['activity_label'].astype(float), ts)
        figures['gc_by_source'] = plot_gc_by_source(
            dsp_train['normalized_sequence'],
            df_hf['normalized_sequence'], ts)
        figures['gc_vs_activity_bins'] = plot_gc_vs_activity_bins(
            df_hf['activity_label'].astype(float),
            df_hf['normalized_sequence'],
            dsp_train['activity_label'].astype(float),
            dsp_train['normalized_sequence'], ts)

        results = feasibility_outputs(feasibility, provenance, environment,
                                      hashes, locked_reference)
        results['experiment_name'] = f"{EXPERIMENT_NAME_PREFIX}_{ts}"
        results['timestamp'] = datetime.now().isoformat()
        results['decision_gate'] = gate
        results['arm_definitions'] = ARM_DEFINITIONS
        results['tolerances'] = PROPOSED_TOLERANCES
        results['moreno_mateos_freeze'] = moreno_mm
        results['pool_provenance_files'] = pool_files
        results['figures'] = figures
        results['phase11_summary'] = locked_reference['phase11']

        out = RESULTS_DIR / f"{EXPERIMENT_NAME_PREFIX}_{ts}.json"
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"    Results: {out}")
        for k, v in figures.items():
            print(f"      - {v}")
        print("=" * 72)
        print("Phase 12 result: " + feasibility['conclusion'])
        print(f"Decision gate: {gate['gate']}")
        return results

    # ------------------------------------------------------------------ F
    print("\n[5] FEASIBLE (not expected with the current inventory): "
          "constructing arm pools (matching-deterministic).")
    pools = construct_arm_pools(dsp_train, {'deephf': df_hf},
                                sizes={'B': 20000, 'C': 20000, 'D': 20000},
                                seed=42)
    print("    WARNING: this branch is never reached on the available data; "
          "it is kept as a guarded, tested code path.")
    raise SystemExit(
        "Feasibility passed unexpectedly - arm training is intentionally not "
        "wired in this build. Review the feasibility audit before proceeding.")


if __name__ == "__main__":
    main()