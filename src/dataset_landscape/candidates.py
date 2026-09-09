"""Public primary-data discovery records for Phase 16A.

Records intentionally remain pending. Phase 16A identifies auditable sources and
provenance questions; later phases must verify files, compatibility, overlap,
and distributions.
"""

from __future__ import annotations

from typing import Dict, List

from .provenance import UNKNOWN, make_record


def candidate_inventory(access_date: str) -> List[Dict[str, object]]:
    """Return deterministic Phase 16A candidate discovery records."""
    records = [
        make_record(
            candidate_id="doench_2016_orcs_publication_screens",
            candidate_name="Doench 2016 published CRISPR screens",
            source_study_id="doench_2016_nbt_3437",
            source_publication=(
                "Doench JG et al. 2016. Optimized sgRNA design to maximize "
                "activity and minimize off-target effects of CRISPR-Cas9. "
                "Nature Biotechnology 34:184-191."
            ),
            source_repository="BioGRID ORCS Dataset 19",
            accession="PubMed:26780180; BioGRID ORCS Dataset:19",
            source_url="https://orcs.thebiogrid.org/Dataset/19",
            access_date=access_date,
            original_filename=(
                "STable_06_Vem_STARSOutputs.xlsx; "
                "STable 12 NegativeSelection_individual_STARS.xlsx; "
                "STable_15_IFNg_data_STARS.xlsx; "
                "STable 09 Sel_STARSOutput.xlsx"
            ),
            local_raw_path=None,
            sha256=None,
            file_format="xlsx",
            parent_dataset=UNKNOWN,
            subset_id="multiple publication supplementary screen tables",
            experimental_condition="A375 and additional screen conditions listed by ORCS",
            derived_or_processed_status="PRIMARY_SUPPLEMENTARY_OR_PUBLISHER_HOSTED",
            assay="CRISPR-Cas9 screens; exact activity-label semantics require 16B/16C verification",
            organism="human/mouse contexts reported by publication; exact rows require verification",
            cell_line_or_system="A375 and other systems reported by source; row-level verification pending",
            Cas9_variant="SpCas9 or publication-specific Cas9; verification pending",
            library_design="publication screen/library design; row-level verification pending",
            sequence_definition="UNKNOWN until original supplementary files are inspected",
            PAM_definition="UNKNOWN until original supplementary files are inspected",
            strand_orientation_documentation="UNKNOWN until original supplementary files are inspected",
            activity_measurement="UNKNOWN; screen outputs listed, label target compatibility unverified",
            activity_units="UNKNOWN",
            continuous_status="UNKNOWN",
            experimental_measured_status="LIKELY_EXPERIMENTAL_SCREEN_OUTPUT_BUT_UNVERIFIED",
            original_preprocessing="UNKNOWN",
            duplicate_policy="UNKNOWN",
            known_subset_or_processed_relationships=(
                "Prior repository Phase 10/15 records suggest possible overlap "
                "with canonical DeepSpCas9 for Doench A375; not reverified in 16A."
            ),
            evidence_notes=[
                "ORCS page lists publication metadata and downloadable supplementary files.",
                "Candidate remains pending because files were not inspected in Phase 16A.",
            ],
        ),
        make_record(
            candidate_id="crispron_2021_rth_tools",
            candidate_name="CRISPRon 2021 sgRNA efficiency data",
            source_study_id="xu_2021_crispron",
            source_publication=(
                "Xu F et al. 2021. Enhancing CRISPR-Cas9 gRNA efficiency "
                "prediction by data integration and deep learning."
            ),
            source_repository="RTH-tools/crispron",
            accession=UNKNOWN,
            source_url="https://github.com/RTH-tools/crispron",
            access_date=access_date,
            original_filename="UNKNOWN; repository files require 16B enumeration",
            local_raw_path=None,
            sha256=None,
            file_format="repository release/source tree; exact data files pending",
            parent_dataset=UNKNOWN,
            subset_id="reported newly generated 10,592 SpCas9 gRNAs plus integrated data",
            experimental_condition="lentiviral surrogate-vector assay; details require file verification",
            derived_or_processed_status="MIXED_PRIMARY_AND_INTEGRATED_DATA_REPORTED",
            assay="CRISPR-Cas9 guide efficiency assay; primary file verification pending",
            organism="human reported; exact row-level organism pending",
            cell_line_or_system="UNKNOWN from 16A repository landing page",
            Cas9_variant="SpCas9 reported",
            library_design="reported SpCas9 gRNA set; library design pending",
            sequence_definition="UNKNOWN until repository data files are inspected",
            PAM_definition="NGG reported by CRISPRon webserver; file-level verification pending",
            strand_orientation_documentation="UNKNOWN",
            activity_measurement="gRNA efficiency reported; exact column definition pending",
            activity_units="UNKNOWN",
            continuous_status="UNKNOWN",
            experimental_measured_status="REPORTED_EXPERIMENTAL_BUT_UNVERIFIED",
            original_preprocessing="UNKNOWN",
            duplicate_policy="UNKNOWN",
            known_subset_or_processed_relationships=(
                "Publication reports integration with complementary published data; "
                "identity and parent/subset relationships require 16B/16C."
            ),
            evidence_notes=[
                "GitHub repository is an official software/data-adjacent source for CRISPRon.",
                "Primary data filenames were not verified in 16A.",
            ],
        ),
        make_record(
            candidate_id="crisprpredseq_2020_bmc_additional_files",
            candidate_name="CRISPRpred(SEQ) additional files HCT116/HEK293/HeLa/HL60",
            source_study_id="rafid_2020_crisprpredseq",
            source_publication=(
                "Rafid AHM et al. 2020. CRISPRpred(SEQ): a sequence-based "
                "method for sgRNA on target activity prediction using "
                "traditional machine learning. BMC Bioinformatics."
            ),
            source_repository="BMC Bioinformatics supplementary additional files",
            accession="DOI:10.1186/s12859-020-3531-9",
            source_url="https://link.springer.com/article/10.1186/s12859-020-3531-9",
            access_date=access_date,
            original_filename="Additional file 1; Additional file 2; Additional file 3; Additional file 4",
            local_raw_path=None,
            sha256=None,
            file_format="publisher supplementary files; exact extensions pending",
            parent_dataset="DeepCRISPR/Haeussler-used on-target dataset reported by article",
            subset_id="HCT116, HEK293, HeLa, HL60 cell-line files",
            experimental_condition="cell-line-specific labeled sgRNA datasets",
            derived_or_processed_status="SECONDARY_REUSE_OR_PROCESSED_VERSION_REPORTED",
            assay="on-target sgRNA activity dataset used by DeepCRISPR; primary origin pending",
            organism="human cell lines reported",
            cell_line_or_system="HCT116; HEK293; HeLa; HL60",
            Cas9_variant="UNKNOWN",
            library_design="UNKNOWN; original dataset design requires source tracing",
            sequence_definition="UNKNOWN until additional files are inspected",
            PAM_definition="UNKNOWN until additional files are inspected",
            strand_orientation_documentation="UNKNOWN",
            activity_measurement="labeled sgRNA activity reported; semantics pending",
            activity_units="UNKNOWN",
            continuous_status="UNKNOWN",
            experimental_measured_status="UNKNOWN; may be processed labels",
            original_preprocessing="UNKNOWN",
            duplicate_policy="UNKNOWN",
            known_subset_or_processed_relationships=(
                "Article explicitly states these data were used in DeepCRISPR "
                "and Haeussler et al.; independence cannot be assumed."
            ),
            evidence_notes=[
                "Publisher page identifies four additional files for HCT116, HEK293, HeLa, HL60.",
                "This is not accepted as independent primary data in 16A.",
            ],
        ),
        make_record(
            candidate_id="deep_hf_2019_public_data_listing",
            candidate_name="DeepHF on-target activity data",
            source_study_id="wang_2019_deephf",
            source_publication=(
                "Wang et al. 2019. Optimized CRISPR guide RNA design for "
                "two high-fidelity Cas9 variants by deep learning."
            ),
            source_repository="DeepHF website/GitHub link reported by public_data_crisprCas9",
            accession=UNKNOWN,
            source_url="https://github.com/dagrate/public_data_crisprCas9",
            access_date=access_date,
            original_filename="UNKNOWN; public_data_crisprCas9 points to DeepHF website/GitHub",
            local_raw_path=None,
            sha256=None,
            file_format="UNKNOWN",
            parent_dataset=UNKNOWN,
            subset_id="SpCas9 and high-fidelity nuclease conditions reported",
            experimental_condition="HEK293T/HCT116/T-cell contexts reported by prior repository records",
            derived_or_processed_status="PRIMARY_SOURCE_LINK_REPORTED_BY_CATALOG; FILE_NOT_VERIFIED",
            assay="indel-rate / on-target activity screens reported; exact source file pending",
            organism="human reported",
            cell_line_or_system="HEK293T/HCT116/T cells reported by prior records; primary verification pending",
            Cas9_variant="SpCas9 and high-fidelity Cas9 variants reported",
            library_design="UNKNOWN",
            sequence_definition="guide+PAM only in prior repository records; primary verification pending",
            PAM_definition="NGG reported in prior repository records; primary verification pending",
            strand_orientation_documentation="UNKNOWN",
            activity_measurement="Wt_Efficiency / indel rate reported by prior records; pending primary verification",
            activity_units="UNKNOWN",
            continuous_status="UNKNOWN",
            experimental_measured_status="REPORTED_EXPERIMENTAL_BUT_UNVERIFIED_IN_16A",
            original_preprocessing="UNKNOWN",
            duplicate_policy="UNKNOWN",
            known_subset_or_processed_relationships=(
                "Previously analyzed in Phase 10; not a new independent candidate "
                "unless primary provenance and intended role are re-approved."
            ),
            evidence_notes=[
                "public_data_crisprCas9 catalog lists DeepHF on-target data and links to data download.",
                "Phase 16A treats this as pending, not compatible or independent.",
            ],
        ),
        make_record(
            candidate_id="sgdesigner_2020_public_data_listing",
            candidate_name="SgDesigner / unique plasmid library sgRNA potency data",
            source_study_id="sgdesigner_2020",
            source_publication="SgDesigner 2020 source publication listed by public_data_crisprCas9",
            source_repository="sgDesigner GitHub link reported by public_data_crisprCas9",
            accession=UNKNOWN,
            source_url="https://github.com/dagrate/public_data_crisprCas9",
            access_date=access_date,
            original_filename="UNKNOWN; source GitHub data file pending",
            local_raw_path=None,
            sha256=None,
            file_format="UNKNOWN",
            parent_dataset=UNKNOWN,
            subset_id=UNKNOWN,
            experimental_condition="human-cell plasmid library context reported by catalog",
            derived_or_processed_status="PRIMARY_SOURCE_LINK_REPORTED_BY_CATALOG; FILE_NOT_VERIFIED",
            assay="quantified CRISPR/Cas9 sgRNA potency; exact assay pending",
            organism="human reported by catalog",
            cell_line_or_system="UNKNOWN",
            Cas9_variant="SpCas9 or CRISPR/Cas9 reported; verification pending",
            library_design="unique plasmid library reported by catalog",
            sequence_definition="UNKNOWN",
            PAM_definition="UNKNOWN",
            strand_orientation_documentation="UNKNOWN",
            activity_measurement="sgRNA potency reported; exact label semantics pending",
            activity_units="UNKNOWN",
            continuous_status="UNKNOWN",
            experimental_measured_status="REPORTED_EXPERIMENTAL_BUT_UNVERIFIED",
            original_preprocessing="UNKNOWN",
            duplicate_policy="UNKNOWN",
            known_subset_or_processed_relationships=UNKNOWN,
            evidence_notes=[
                "Catalog lists SgDesigner as an on-target dataset with data-download link.",
                "Original repository and file must be verified before compatibility decisions.",
            ],
        ),
    ]
    return sorted(records, key=lambda item: str(item["candidate_id"]))
