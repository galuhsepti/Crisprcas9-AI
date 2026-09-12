# Phase 17G - Data Sufficiency and Readiness Assessment

## 1. Why Phase 17G Exists

Phase 17G defines an objective, configurable GO / NO-GO framework for deciding when recovered public CRISPR-Cas9 datasets are sufficiently usable to justify a future modeling phase. It is an audit-only readiness assessment, not a model-improvement experiment.

## 2. Why Not Phase 18

Phase 17F accepted no dataset for future pipeline integration. Phase 17G therefore records what would be required before moving beyond data recovery; it does not train, evaluate, tune, pool, or integrate any dataset.

## 3. Hard Scientific Compatibility Requirements

- Verifiable primary experimental source.
- Compatible SpCas9 on-target task evidence.
- Directly supported sequence geometry; no constructed 30-mers or inferred flanks.
- Explicit continuous experimental activity label; no binary, rank, enrichment, log-fold-change, STARS, classifier-label, or score rescue.
- Acceptable independence/provenance relationship without unresolved derivation/leakage concern.

## 4. Project-Specific Policy Thresholds

- minimum_candidate_unique_n = 1000
- recommended_candidate_unique_n = 2000
- minimum_total_new_unique_n = 2000
- minimum_continuous_label_unique_values = 20

These are conservative project-management thresholds relative to the DeepSpCas9 canonical modeling population of 10,117. They are not universal CRISPR biology or machine-learning laws.

## 5. Candidate Results

| candidate_id | provenance | task | sequence valid n | unique seq n | label status | independence | quantity | information | final candidate status |
|---|---|---|---:|---:|---|---|---|---|---|
| `corsi_2022_free_energy_pam_context` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | 0 | 0 | `UNKNOWN_SCALE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_QUANTITY` | `INSUFFICIENT_EVIDENCE` | `NOT_READY` |
| `crispron_2021_rth_tools` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | 0 | 0 | `UNKNOWN_SCALE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_QUANTITY` | `INSUFFICIENT_EVIDENCE` | `NOT_READY` |
| `crisprpredseq_2020_bmc_additional_files` | `PASS` | `INCOMPATIBLE` | 16749 | 12465 | `INCOMPATIBLE_TARGET` | `FAIL` | `INSUFFICIENT_QUANTITY` | `LOW` | `NOT_READY` |
| `deep_hf_2019_public_data_listing` | `FAIL` | `INSUFFICIENT_EVIDENCE` | 0 | 0 | `UNKNOWN_SCALE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_QUANTITY` | `INSUFFICIENT_EVIDENCE` | `NOT_READY` |
| `doench_2016_orcs_publication_screens` | `PASS` | `INCOMPATIBLE` | 194653 | 157573 | `INCOMPATIBLE_TARGET` | `FAIL` | `INSUFFICIENT_QUANTITY` | `LOW` | `NOT_READY` |
| `sgdesigner_2020_public_data_listing` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_EVIDENCE` | 0 | 0 | `UNKNOWN_SCALE` | `INSUFFICIENT_EVIDENCE` | `INSUFFICIENT_QUANTITY` | `INSUFFICIENT_EVIDENCE` | `NOT_READY` |

## 6. Aggregate Accepted New Data

accepted_candidate_count = 0

total_new_unique_compatible_observations = 0

additional_compatible_unique_observations_required = 2000

## 7. Final Decision

NO_GO

## 8. Blocking Reasons

- No Phase 17 candidate satisfies provenance, compatibility, label, independence, quantity, and information-value gates.
- `corsi_2022_free_energy_pam_context`: Provenance gate did not pass.
- `corsi_2022_free_energy_pam_context`: Biological/task compatibility gate did not pass.
- `corsi_2022_free_energy_pam_context`: Compatible continuous label quality was not established.
- `corsi_2022_free_energy_pam_context`: Independence/leakage gate did not pass.
- `corsi_2022_free_energy_pam_context`: Project-specific unique compatible observation threshold was not met.
- `crispron_2021_rth_tools`: Provenance gate did not pass.
- `crispron_2021_rth_tools`: Biological/task compatibility gate did not pass.
- `crispron_2021_rth_tools`: Compatible continuous label quality was not established.
- `crispron_2021_rth_tools`: Independence/leakage gate did not pass.
- `crispron_2021_rth_tools`: Project-specific unique compatible observation threshold was not met.
- `crisprpredseq_2020_bmc_additional_files`: Biological/task compatibility gate did not pass.
- `crisprpredseq_2020_bmc_additional_files`: Compatible continuous label quality was not established.
- `crisprpredseq_2020_bmc_additional_files`: Independence/leakage gate did not pass.
- `crisprpredseq_2020_bmc_additional_files`: Project-specific unique compatible observation threshold was not met.
- `deep_hf_2019_public_data_listing`: Provenance gate did not pass.
- `deep_hf_2019_public_data_listing`: Biological/task compatibility gate did not pass.
- `deep_hf_2019_public_data_listing`: Compatible continuous label quality was not established.
- `deep_hf_2019_public_data_listing`: Independence/leakage gate did not pass.
- `deep_hf_2019_public_data_listing`: Project-specific unique compatible observation threshold was not met.
- `doench_2016_orcs_publication_screens`: Biological/task compatibility gate did not pass.
- `doench_2016_orcs_publication_screens`: Compatible continuous label quality was not established.
- `doench_2016_orcs_publication_screens`: Independence/leakage gate did not pass.
- `doench_2016_orcs_publication_screens`: Project-specific unique compatible observation threshold was not met.
- `sgdesigner_2020_public_data_listing`: Provenance gate did not pass.
- `sgdesigner_2020_public_data_listing`: Biological/task compatibility gate did not pass.
- `sgdesigner_2020_public_data_listing`: Compatible continuous label quality was not established.
- `sgdesigner_2020_public_data_listing`: Independence/leakage gate did not pass.
- `sgdesigner_2020_public_data_listing`: Project-specific unique compatible observation threshold was not met.

## 9. What Would Change NO-GO Into GO

A future GO would require recovered datasets with primary experimental provenance, compatible SpCas9 on-target 30-mer-or-defensibly-preprocessable sequence evidence, explicit continuous activity labels with at least 20 observed values, acceptable independence, at least 1,000 unique compatible observations per accepted candidate, and at least 2,000 total newly accepted unique compatible observations.

## 10. Scope Confirmation

- No model training occurred.
- No model evaluation occurred.
- No Moreno evaluation or raw Moreno access occurred.
- No dataset pooling occurred.
- No label or sequence transformation occurred.
- Phase 18 was not started.

## 11. Canonical Integrity

canonical_integrity_unchanged = True
