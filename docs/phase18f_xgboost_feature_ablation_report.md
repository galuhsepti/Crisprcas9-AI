# Phase 18F - Locked XGBoost Feature-Ablation Experiment

## Completion

Status: **COMPLETED**

Registered fits: **18/18 completed**, 0 failed, 0 skipped.

Prediction bundles: **18/18 completed**.

This experiment measures predictive information contribution, not biological causality, molecular necessity, or Cas9 mechanism.

## Full Controls

| Domain | Spearman | RMSE | Pearson | R2 | MAE |
|---|---:|---:|---:|---:|---:|
| deepspcas9 | 0.70804937 | 0.15501222 | 0.71457965 | 0.50746017 | 0.12277021 |
| crispron_xiang_luo | 0.75578064 | 15.51928110 | 0.75077910 | 0.55667142 | 12.15414820 |

## Drop-One Results

| Domain | Family removed | Delta Spearman | Delta RMSE | Relative RMSE degradation | Classification |
|---|---|---:|---:|---:|---|
| deepspcas9 | GC_AND_SKEW | -0.00320964 | 0.00065881 | 0.00425008 | LITTLE_UNIQUE_CONTRIBUTION |
| deepspcas9 | NUCLEOTIDE_COMPOSITION | 0.00103974 | -0.00006045 | -0.00039000 | LITTLE_UNIQUE_CONTRIBUTION |
| deepspcas9 | DINUCLEOTIDE_FREQUENCY | -0.00002808 | 0.00001600 | 0.00010319 | LITTLE_UNIQUE_CONTRIBUTION |
| deepspcas9 | GLOBAL_KMER | -0.01179359 | 0.00174539 | 0.01125971 | LITTLE_UNIQUE_CONTRIBUTION |
| deepspcas9 | ENTROPY_COMPLEXITY | -0.00050614 | 0.00002274 | 0.00014669 | LITTLE_UNIQUE_CONTRIBUTION |
| deepspcas9 | POSITION_SPECIFIC_NUCLEOTIDE | -0.24985138 | 0.03889015 | 0.25088442 | ESSENTIAL_OR_STRONG_CONTRIBUTOR |
| crispron_xiang_luo | GC_AND_SKEW | -0.00818101 | 0.15356703 | 0.00989524 | LITTLE_UNIQUE_CONTRIBUTION |
| crispron_xiang_luo | NUCLEOTIDE_COMPOSITION | 0.00012977 | 0.01954081 | 0.00125913 | LITTLE_UNIQUE_CONTRIBUTION |
| crispron_xiang_luo | DINUCLEOTIDE_FREQUENCY | -0.00003813 | 0.00077618 | 0.00005001 | LITTLE_UNIQUE_CONTRIBUTION |
| crispron_xiang_luo | GLOBAL_KMER | 0.00388550 | -0.09291156 | -0.00598685 | LITTLE_UNIQUE_CONTRIBUTION |
| crispron_xiang_luo | ENTROPY_COMPLEXITY | 0.00076183 | -0.01375165 | -0.00088610 | LITTLE_UNIQUE_CONTRIBUTION |
| crispron_xiang_luo | POSITION_SPECIFIC_NUCLEOTIDE | -0.29311055 | 5.06043853 | 0.32607429 | ESSENTIAL_OR_STRONG_CONTRIBUTOR |

## Domain Consistency

| Family | Classification |
|---|---|
| GC_AND_SKEW | WEAK_IN_BOTH |
| NUCLEOTIDE_COMPOSITION | WEAK_IN_BOTH |
| DINUCLEOTIDE_FREQUENCY | WEAK_IN_BOTH |
| GLOBAL_KMER | WEAK_IN_BOTH |
| ENTROPY_COMPLEXITY | WEAK_IN_BOTH |
| POSITION_SPECIFIC_NUCLEOTIDE | CROSS_DOMAIN_CONSISTENT |

## Family-Only Models

| Domain | Family | Spearman | RMSE | Spearman/full | RMSE/full |
|---|---|---:|---:|---:|---:|
| deepspcas9 | POSITION_SPECIFIC_NUCLEOTIDE | 0.70080518 | 0.15664998 | 0.98976881 | 1.01056538 |
| deepspcas9 | GLOBAL_KMER | 0.44135143 | 0.19601848 | 0.62333426 | 1.26453568 |
| crispron_xiang_luo | POSITION_SPECIFIC_NUCLEOTIDE | 0.73762972 | 15.89697362 | 0.97598388 | 1.02433699 |
| crispron_xiang_luo | GLOBAL_KMER | 0.45129858 | 20.75425287 | 0.59712905 | 1.33732051 |

## Interpretation Boundary

Small drop-one effects indicate little unique contribution conditional on retained representations; they do not show that a feature family is useless. GC, composition, dinucleotide, k-mer, complexity, and positional representations overlap.

Domains retain native label scales and are not pooled. Bridge and quarantine observations were excluded. The locked external dataset was not accessed.

## Limitations

- Results use one fixed Phase 18C split and two development domains.
- The deterministic fit policy does not estimate model-fit variance.
- Family ablations cannot isolate information shared redundantly by multiple retained families.
- Internal-test evidence does not establish external generalization.
- Statistical support is not biological causality.

## Execution Recovery

The 18 fits and deterministic prediction writes completed under the confirmed training implementation before the orchestration process exceeded its tool time limit during bootstrap analysis. Recovery performed analysis, bootstrap, and reporting only: `additional_fits = 0` and `additional_predictions = 0`. No model was refit and no prediction was regenerated. All 10,000 paired PCG64(42) bootstrap replicates per domain were completed from the frozen prediction bundles using the locked statistical definitions. Process-local wall-clock and tracemalloc measurements were unavailable after termination; each saved model retains its exact XGBoost-reported training time. Moreno-Mateos remained untouched and was not accessed.
