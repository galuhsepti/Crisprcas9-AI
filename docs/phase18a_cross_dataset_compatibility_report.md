# Phase 18A - Cross-Dataset Label Compatibility and Integration Strategy Audit

## Scope

- Audit only; no model training, retraining, tuning, calibration, or evaluation occurred.
- No label normalization, transformation, harmonization fit, or training dataset creation occurred.
- The locked external dataset was not accessed.
- Canonical model code, data, artifacts, and evaluation policy remain unchanged.

## Primary Answer

- Immediate direct pooling justified: **False**
- Controlled integration experiments justified: **True**
- Primary recommendation: **MULTI_DOMAIN_EXPERIMENT_RECOMMENDED**

Phase 17G-R1 GO establishes data sufficiency for controlled experiments. It does not establish that the two response variables are directly exchangeable.

## Label Semantics

### deepspcas9

- Publication: Kim et al. 2019, Science Advances 5:eaax9249 (10.1126/sciadv.aax9249)
- Experimental assay: Pooled lentiviral guide-plus-integrated-surrogate-target library followed by targeted deep sequencing
- Cell type(s): HEK293T
- Target system: Integrated 30-nt surrogate target paired with its sgRNA; not an endogenous-locus training measurement
- Cas nuclease: SpCas9
- Label: `activity (source derivative: modFreq)`
- Activity definition: Fraction of reads with an insertion or deletion in the 8-nt window centered on the expected cleavage site
- Raw biological measurement: Day-2.9 indel reads divided by total reads
- Normalization: Local values equal the source table's uncorrected indel percentage divided by 100; the publication separately describes background-corrected indel frequency
- Numerical unit: fraction
- Higher means greater editing efficiency: True
- Direct primary measurement: True
- Original-author transformations: Publication background-correction formula exists, but the project-local activity column traces to the uncorrected indel-frequency field rather than that corrected field
- Timing: 2.9 days after SpCas9 lentiviral transduction
- Evidence caveat: The exact response column used by the original authors' final model is not fully established from the published training code; this audit describes the repository-local label actually used by this thesis pipeline.
- Evidence chain:
  1. Kim et al. 2019 primary publication and Table S1 (HT_Cas9_Train)
  2. third-party Benchmarking-CRISPR-on-tools DeepSpCas9 (Library) derivative
  3. scripts/prepare_data.py renames modFreq to activity without label transformation
  4. data/raw/DeepSpCas9.csv activity

### crispron_xiang_luo

- Publication: Xiang et al. 2021, Nature Communications 12:3238 (10.1038/s41467-021-23576-0)
- Experimental assay: Pooled lentiviral 37-bp surrogate-target library followed by targeted amplicon deep sequencing
- Cell type(s): HEK293T-SpCas9 stable line, wild-type HEK293T background control
- Target system: Randomly integrated 37-bp surrogate target containing 10-nt upstream, spacer, PAM, and 4-nt downstream context
- Cas nuclease: Human-codon-optimized SpCas9
- Label: `HEK293T_indel_freq_avg_d8_d10`
- Activity definition: Mean Dox-free total indel efficiency at days 8 and 10 after guide/surrogate transduction
- Raw biological measurement: Percentage of amplicon reads classified as CRISPR-induced indels after removing background sequence variants, averaged across days 8 and 10
- Normalization: No Phase 18A normalization; workbook-native raw day-8/day-10 average is audited. The publication later linearly rescales Xiang/Luo values for its own Kim-integrated training dataset.
- Numerical unit: percent
- Higher means greater editing efficiency: True
- Direct primary measurement: True
- Original-author transformations: Arithmetic day-8/day-10 averaging for this field; low-read filtering and background-variant removal. A separate integrated-label field applies linear cross-study rescaling and is not used here.
- Timing: 8 and 10 days after guide/surrogate transduction, Dox-free, with puromycin selection from day 2
- Evidence caveat: Official training documentation references Quant_norm_efficiency, which is absent from the recovered workbook; no identity or transformation for that field is assumed.
- Evidence chain:
  1. Xiang et al. 2021 primary publication and Supplementary Data 1
  2. official CRISPRon/RTH Luo2020_Kim2019.xlsx download
  3. Dataset values LuoSpCas92020_min200 plus Overlap_Luo_Kim2019
  4. 30mer_gRNA
  5. HEK293T_indel_freq_avg_d8_d10

### Concept Comparison

- Same underlying biological concept: **True**
- Concept: Sequence-linked SpCas9 on-target indel efficiency in HEK293T surrogate-target assays
- Operationally equivalent endpoints: **False**
- Reason: Units, timing, selection, Cas9 expression context, and background handling differ materially.

## Label Scale Comparison

- Numerical scales directly comparable: **False**
- No values were converted, normalized, ranked, or otherwise transformed.

| statistic | DeepSpCas9 canonical n=10,117 | Xiang/Luo n=10,592 |
|---|---:|---:|
| n | 10117 | 10592 |
| finite_n | 10117 | 10592 |
| missing_n | 0 | 0 |
| min | 0.000000 | 0.038023 |
| max | 0.951350 | 95.851258 |
| mean | 0.425652 | 37.160105 |
| median | 0.436127 | 35.139751 |
| SD | 0.220414 | 22.993814 |
| IQR | 0.353029 | 38.112750 |
| unique_value_n | 10097 | 10592 |
| skewness | -0.062065 | 0.305302 |
| fraction_at_min | 0.001384 | 0.000094 |
| fraction_at_max | 0.000099 | 0.000094 |

Quantiles:

| percentile | DeepSpCas9 | Xiang/Luo |
|---:|---:|---:|
| 0 | 0.000000 | 0.038023 |
| 1 | 0.009056 | 1.872812 |
| 5 | 0.060595 | 4.988472 |
| 10 | 0.115953 | 7.958040 |
| 25 | 0.249543 | 16.923864 |
| 50 | 0.436127 | 35.139751 |
| 75 | 0.602572 | 55.036614 |
| 90 | 0.714389 | 69.867297 |
| 95 | 0.770959 | 77.188056 |
| 99 | 0.853903 | 86.617927 |
| 100 | 0.951350 | 95.851258 |

Different numeric ranges do not imply biological incompatibility, but the unit and endpoint differences prevent treating values as exchangeable regression targets without a controlled study.

## Shared-Sequence Bridge

- DeepSpCas9 bridge population: **raw provenance population (n=12832)**
- Exact shared 30-mers: **48**
- Finite label pairs: 48
- Pearson r: 0.675358 (p=0.000000)
- Spearman rho: 0.687799 (p=0.000000)
- Kendall tau: 0.503546 (p=0.000000)
- Monotonic relationship: MODERATE_POSITIVE_MONOTONIC_ASSOCIATION
- Mean absolute percentile-rank disagreement: 0.177305
- Same-quartile fraction: 0.500000
- Linear slope/intercept (diagnostic only): 78.585185 / 3.860256
- Theil-Sen slope/intercept (diagnostic only): 85.490278 / -0.062843
- Regression outliers: 0

Raw numeric disagreement is reported in JSON but is not interpreted because one source uses fractions and the other percentages. The bridge was not used to fit a future-training harmonization transform.

### Canonical-Population Sensitivity

- DeepSpCas9 canonical population: n=10117
- Exact shared 30-mers: 41
- Pearson r: 0.744308
- Spearman rho: 0.746690
- Relationship: STRONG_POSITIVE_MONOTONIC_ASSOCIATION
- Interpretation: Sensitivity analysis only; it does not replace the all-overlap provenance bridge or authorize label harmonization.

| 30-mer | DeepSpCas9 | Xiang/Luo | Deep rank | Xiang rank | abs rank disagreement | outlier |
|---|---:|---:|---:|---:|---:|---|
| `AAATTGGAGGCCAAAGCCTTAATCTGGACT` | 0.197248 | 7.723451 | 0.063830 | 0.063830 | 0.000000 | False |
| `AACGTCTTGCTCGAGATGTGATGAAGGAGA` | 0.657566 | 38.846630 | 0.808511 | 0.510638 | 0.297872 | False |
| `AAGAGGAACCAGTGCCGCTACTGCAGGCTC` | 0.534992 | 78.211166 | 0.638298 | 0.936170 | 0.297872 | False |
| `AAGCTCACAGACAGGTGATGCACTTGGAAA` | 0.427928 | 19.972684 | 0.468085 | 0.255319 | 0.212766 | False |
| `AAGTTGTTCCAGATCCAGTTCAACCGGAGT` | 0.230341 | 14.694309 | 0.148936 | 0.212766 | 0.063830 | False |
| `AATGAGCCAGGCAGCCCGCATCACAGGCAC` | 0.402324 | 24.257743 | 0.404255 | 0.319149 | 0.085106 | False |
| `ACCAACTGAGCCAACATGTGACCGTGGACG` | 0.807100 | 64.538000 | 0.978723 | 0.808511 | 0.170213 | False |
| `ACGCTATCGCCCAGGGCGGGGAGGAGGAGT` | 0.194527 | 10.284691 | 0.042553 | 0.106383 | 0.063830 | False |
| `ACGCTTCCAGAACGCCTGCCGCGACGGCCG` | 0.540418 | 54.928207 | 0.680851 | 0.702128 | 0.021277 | False |
| `ACGTCCTCTACCCACATGTGCATACGGAGG` | 0.207692 | 9.575671 | 0.085106 | 0.085106 | 0.000000 | False |
| `ACTTGGTACAATGGACCCGACACATGGAGG` | 0.470335 | 48.215058 | 0.531915 | 0.574468 | 0.042553 | False |
| `AGCCCTTGTGCCAGTTGCTCTTGCAGGTGT` | 0.257130 | 2.603248 | 0.212766 | 0.000000 | 0.212766 | False |
| `CAACAACTTCCAGGGGCGCTACGATGGCAA` | 0.513010 | 57.100763 | 0.617021 | 0.723404 | 0.106383 | False |
| `CAGCTGACAACCAGGAGAAGATCGGGGGTG` | 0.787515 | 67.337187 | 0.936170 | 0.851064 | 0.085106 | False |
| `CAGCTGCTACACAATTAGGACTGAAGGAGA` | 0.359319 | 22.744373 | 0.382979 | 0.297872 | 0.085106 | False |
| `CAGGAGCCCAGCAATTTGTGGCAAAGGATC` | 0.311046 | 16.342580 | 0.276596 | 0.234043 | 0.042553 | False |
| `CAGGATTAGCCGATCGTTACCTCAAGGGAG` | 0.634678 | 63.986392 | 0.787234 | 0.787234 | 0.000000 | False |
| `CCAGGAAACAGGGGCTGGTGGCATGGGTGG` | 0.130144 | 51.969935 | 0.021277 | 0.659574 | 0.638298 | False |
| `CCATGAAGCAGTTTCTGCTGTACTTGGATG` | 0.218941 | 13.066885 | 0.106383 | 0.191489 | 0.085106 | False |
| `CCTATTTCAGACACAGTTAAGCTCTGGAAC` | 0.693820 | 50.625907 | 0.851064 | 0.617021 | 0.234043 | False |
| `CCTGACCTGTCAGGTGAAGTTCGCTGGAGC` | 0.247851 | 27.463318 | 0.191489 | 0.404255 | 0.212766 | False |
| `CTGAGCAGCAGGCTGCACTCCTGCTGGCAT` | 0.352827 | 32.601030 | 0.361702 | 0.446809 | 0.085106 | False |
| `CTGCAGTTCTACGAATCCGTTCAGTGGAGG` | 0.735584 | 52.684924 | 0.872340 | 0.680851 | 0.191489 | False |
| `CTTCCCACTCGAGGCCAGCTTGAAGGGAGA` | 0.270784 | 5.738222 | 0.234043 | 0.042553 | 0.191489 | False |
| `GACAGATCCAGATCCAGGGTGGACAGGCTG` | 0.498950 | 26.373457 | 0.574468 | 0.361702 | 0.212766 | False |
| `GACGACCCCAGGGTAACTCCCGGCAGGTGG` | 0.348049 | 62.136222 | 0.340426 | 0.765957 | 0.425532 | False |
| `GAGGTCGGTGCAGCTCCACGACTCTGGAAA` | 0.747967 | 24.914530 | 0.893617 | 0.340426 | 0.553191 | False |
| `GAGGTTCCAGGAGCCCCCGCCTGGAGGAGC` | 0.483289 | 60.831097 | 0.553191 | 0.744681 | 0.191489 | False |
| `GCAACGAGCAGCGCACGCTCATCAAGGCCT` | 0.320345 | 10.648351 | 0.319149 | 0.127660 | 0.191489 | False |
| `GCAATGCTTTCCAACATTGATGAGTGGATT` | 0.805819 | 44.200456 | 0.957447 | 0.531915 | 0.425532 | False |
| `GCAGCTGACAACCAGGAGAAGATCGGGGGT` | 0.434563 | 26.462638 | 0.510638 | 0.382979 | 0.127660 | False |
| `GCCGGGGCAGCCGGTGCCCGAAATTGGGCT` | 0.022458 | 5.636645 | 0.000000 | 0.021277 | 0.021277 | False |
| `GCTAGCAAGCGAAGCTACCAGTTCTGGGAT` | 0.278813 | 12.369007 | 0.255319 | 0.148936 | 0.106383 | False |
| `GGAGCAAGAGCCGATCTCCTCACAAGGGAG` | 0.511614 | 51.305489 | 0.595745 | 0.638298 | 0.042553 | False |
| `GGCACAGGCCAGGTCATCAAGGGCTGGGAC` | 0.245964 | 28.436843 | 0.170213 | 0.425532 | 0.255319 | False |
| `GGCCAAAGCCACAAGAATCCGCACAGGGTT` | 0.625364 | 69.037201 | 0.744681 | 0.872340 | 0.127660 | False |
| `GGGGCATCCAGGCGCACACTGCCGAGGAGG` | 0.692790 | 66.642503 | 0.829787 | 0.829787 | 0.000000 | False |
| `GTACTTCCAGACCGTGACTGACTATGGCAA` | 0.225674 | 13.003792 | 0.127660 | 0.170213 | 0.042553 | False |
| `GTCTCCTTCCGAAAACACTACCCTTGGGTC` | 0.537909 | 22.577500 | 0.659574 | 0.276596 | 0.382979 | False |
| `TCCAGGAAACAGGGGCTGGTGGCATGGGTG` | 0.317232 | 34.541262 | 0.297872 | 0.468085 | 0.170213 | False |
| `TGAGAAGCAGAACTTACTATCCGTTGGCGA` | 0.610107 | 76.317144 | 0.702128 | 0.893617 | 0.191489 | False |
| `TGAGGACTGCCACGGCGTACTCTGAGGAGT` | 0.408040 | 79.180710 | 0.425532 | 0.957447 | 0.531915 | False |
| `TGCCAGCACCAGTTCCGCGGCCGCCGGTGG` | 0.632101 | 80.649120 | 0.765957 | 1.000000 | 0.234043 | False |
| `TGGACCGGCACCAGTACAACTACGTGGACG` | 0.853692 | 76.739534 | 1.000000 | 0.914894 | 0.085106 | False |
| `TGGTCAGGCAGTATAATCCAAAGATGGTCA` | 0.775616 | 44.777873 | 0.914894 | 0.553191 | 0.361702 | False |
| `TGGTGCTTCCAGCGGGTCTGTATGTGGGGA` | 0.423547 | 35.168868 | 0.446809 | 0.489362 | 0.042553 | False |
| `TGTGTCCCAGACGCAGCACGCCAGTGGTCA` | 0.620445 | 79.932708 | 0.723404 | 0.978723 | 0.255319 | False |
| `TTATGCTTCAGACTCTCAACGATGAGGTTC` | 0.432684 | 49.611965 | 0.489362 | 0.595745 | 0.106383 | False |

## Biological Domains

| factor | DeepSpCas9 | Xiang/Luo | severity |
|---|---|---|---|
| cell_type | HEK293T | Stable HEK293T-SpCas9 line plus wild-type HEK293T background control | MODERATE |
| assay_protocol | Paired guide/30-nt target lentiviral library; Cas9 delivered after cell-library generation | Paired guide/37-bp surrogate lentiviral library transduced into stable Cas9 cells | MAJOR |
| Cas9_expression_context | SpCas9 lentivirus, MOI 5, blasticidin selection | PiggyBac-integrated TRE-SpCas9 stable line; basal Dox-free expression | MAJOR |
| experimental_platform | Lentiviral integrated surrogate and targeted deep sequencing | Lentiviral integrated surrogate and targeted amplicon sequencing | MINOR |
| gene_target_composition | GeCKOv1, selected human/mouse genes, drug-resistance genes, synthetic extreme-GC targets, and laboratory targets | Early coding exons from approximately 3,832-3,834 human protein-coding genes after off-target/library filters | MAJOR |
| measurement_timing | Day 2.9 after Cas9 delivery | Mean of Dox-free days 8 and 10 after guide/surrogate delivery | MAJOR |
| sequence_library_construction | 15,656 designed 30-nt targets in paired guide-target oligos | 12K oligo pool with 37-bp surrogate targets and third-generation lentiviral vector | MODERATE |
| target_locus_composition | Human-derived, mouse-derived, and synthetic target sequences measured at integrated surrogates | Human coding-exon-derived sequences measured at randomly integrated surrogates | MAJOR |
| endogenous_vs_synthetic_context | Training labels from integrated surrogate targets | Training labels from randomly integrated surrogate targets | MINOR |
| background_handling | Project-local label traces to uncorrected indel fraction; publication also reports a background-corrected endpoint | Wild-type controls used to remove background variants before total-indel estimation | MAJOR |

Major differences: assay_protocol, Cas9_expression_context, gene_target_composition, measurement_timing, target_locus_composition, background_handling.

Unknown metadata:
- Exact DeepSpCas9 Cas9-vector promoter and expression level in the assayed cells
- Exact equivalence between CRISPRon Quant_norm_efficiency and any recovered workbook field

## Sequence Domains

- DeepSpCas9 population: canonical modeling n=10117
- Xiang/Luo population: n=10592
- Mean 30-mer GC delta (Xiang/Luo minus DeepSpCas9): 0.050058
- Mean spacer GC delta: 0.053711
- PAM JSD (bits): 0.034549
- Mean/max positional nucleotide JSD (bits): 0.009840 / 0.036518
- Full 30-mer k=2/k=3 JSD: 0.008480 / 0.016648
- Spacer k=2/k=3 JSD: 0.014908 / 0.030294
- Interpretation: Sequence distributions differ descriptively; these differences are not treated as biological incompatibility or a PASS/FAIL gate.

Full positional frequencies, PAM distributions, k-mer summaries, entropy, and complexity descriptors are retained in the JSON artifact.

## Integration Strategies

| strategy | classification | central assessment |
|---|---|---|
| A DIRECT_POOLING | **NOT_RECOMMENDED** | Low under current evidence because equal numeric targets would imply endpoint and unit equivalence not established by this audit. |
| B WITHIN-DATASET NORMALIZATION THEN POOLING | **POSSIBLE_WITH_CAUTION** | Can remove dataset-level location/scale differences but cannot remove biological timing or assay effects. |
| C RANK-BASED INTEGRATION | **POSSIBLE_WITH_CAUTION** | Defensible for guide prioritization if relative ordering is the declared target. |
| D MULTI-DOMAIN LEARNING | **RECOMMENDED** | Best preserves both observed endpoints while explicitly representing assay identity. |
| E TRANSFER LEARNING | **POSSIBLE_WITH_CAUTION** | Plausible because both domains assay SpCas9 indels, but adaptation direction and validation policy must be pre-registered. |
| F EXTERNAL DEVELOPMENT DOMAIN | **RECOMMENDED** | Strong conservative comparator that tests cross-domain development without asserting label exchangeability. |

Detailed assumptions and risks for each strategy are in the JSON artifact.

## Decision Reasons

- Phase 17G-R1 GO establishes sufficient accepted new data, not direct label exchangeability.
- Both labels measure sequence-linked SpCas9 indel efficiency in HEK293T integrated-surrogate assays.
- DeepSpCas9 activity is a day-2.9 fraction while Xiang/Luo is a Dox-free day-8/day-10 mean percentage.
- Cas9 expression, selection duration, background handling, and library/target composition differ materially.
- The raw-population bridge contains all 48 exact shared 30-mers; the canonical-population sensitivity contains 41. Neither authorizes a harmonization transform in Phase 18A.
- Multi-domain experiments can preserve original labels and dataset identity while testing shared versus domain-specific signal.
- A separate Xiang/Luo development-domain arm should be retained as a conservative comparator.

## Canonical Integrity

- Safe-file hashes unchanged during audit: True
- All expected safe-file hashes match: True
- Canonical data modified: False
- Canonical model code modified: False
- Canonical artifacts modified: False
- Locked external data accessed: False

## Stop

Phase 18B was not started. Approval is required before any integration experiment or label transformation.
