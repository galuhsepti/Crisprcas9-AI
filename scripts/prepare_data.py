#!/usr/bin/env python3
"""
Prepare CRISPR-Cas9 sgRNA datasets for training.

This script loads the raw data from the Benchmarking repository,
performs basic validation, and saves clean versions for the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_deepspcas9(data_dir: Path) -> pd.DataFrame:
    """Load DeepSpCas9 dataset."""
    filepath = data_dir / 'external/Benchmarking-CRISPR-on-tools/Training/DeepSpCas9 (Library).csv'
    df = pd.read_csv(filepath)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'sequence23': 'sequence_23mer',
        'sequence30': 'sequence_30mer',
        'modFreq': 'activity'
    })
    
    # Add ID column
    df['id'] = [f'DeepSpCas9_{i:05d}' for i in range(len(df))]
    
    # Extract guide sequence (positions 4-23 from 30-mer)
    df['guide_sequence'] = df['sequence_30mer'].str[4:24]
    
    # Extract PAM (positions 24-26)
    df['pam'] = df['sequence_30mer'].str[24:27]
    
    # Calculate GC content in guide
    df['gc_content'] = df['guide_sequence'].apply(
        lambda x: (x.count('G') + x.count('C')) / 20
    )
    
    logger.info(f"Loaded DeepSpCas9: {len(df)} samples")
    return df


def load_moreno_mateos(data_dir: Path) -> pd.DataFrame:
    """Load Moreno-Mateos dataset for independent testing."""
    filepath = data_dir / 'external/Benchmarking-CRISPR-on-tools/Training/Moreno-Mateos.csv'
    df = pd.read_csv(filepath)
    
    # Rename columns
    df = df.rename(columns={
        'sequence23': 'sequence_23mer',
        'sequence30': 'sequence_30mer',
        'modFreq': 'activity'
    })
    
    # Add ID column
    df['id'] = [f'MorenoMateos_{i:05d}' for i in range(len(df))]
    
    # Extract guide sequence
    df['guide_sequence'] = df['sequence_30mer'].str[4:24]
    
    # Extract PAM
    df['pam'] = df['sequence_30mer'].str[24:27]
    
    # Calculate GC content
    df['gc_content'] = df['guide_sequence'].apply(
        lambda x: (x.count('G') + x.count('C')) / 20
    )
    
    logger.info(f"Loaded Moreno-Mateos: {len(df)} samples")
    return df


def validate_dataset(df: pd.DataFrame, name: str) -> bool:
    """Validate dataset quality."""
    logger.info(f"Validating {name}...")
    
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
    
    # Check sequence lengths
    if (df['sequence_30mer'].str.len() != 30).any():
        issues.append("Invalid 30-mer lengths found")
    
    if (df['guide_sequence'].str.len() != 20).any():
        issues.append("Invalid guide lengths found")
    
    # Check PAM sequences
    valid_pam = df['pam'].str.match(r'^[ACGT]GG$')
    if not valid_pam.all():
        invalid_count = (~valid_pam).sum()
        issues.append(f"Invalid PAM sequences: {invalid_count}")
    
    # Check nucleotides
    for col in ['sequence_30mer', 'guide_sequence']:
        unique_chars = set(''.join(df[col].values))
        invalid_chars = unique_chars - {'A', 'C', 'G', 'T'}
        if invalid_chars:
            issues.append(f"Invalid characters in {col}: {invalid_chars}")
    
    # Check activity range
    if df['activity'].min() < 0 or df['activity'].max() > 1:
        issues.append(f"Activity out of [0,1] range: [{df['activity'].min()}, {df['activity'].max()}]")
    
    # Check duplicates
    if df['sequence_30mer'].duplicated().any():
        dup_count = df['sequence_30mer'].duplicated().sum()
        issues.append(f"Duplicate sequences: {dup_count}")
    
    if issues:
        logger.warning(f"Validation issues for {name}:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info(f"{name} validation passed")
        return True


def save_dataset(df: pd.DataFrame, output_path: Path, description: str):
    """Save dataset with metadata."""
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {description}: {output_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix('.metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Dataset: {description}\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n")
        f.write(f"Activity range: [{df['activity'].min():.4f}, {df['activity'].max():.4f}]\n")
        f.write(f"Activity mean: {df['activity'].mean():.4f}\n")
        f.write(f"Activity std: {df['activity'].std():.4f}\n")


def main():
    """Main function."""
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("CRISPR Dataset Preparation")
    logger.info("=" * 60)
    
    # Load datasets
    logger.info("\n1. Loading datasets...")
    deepspcas9_df = load_deepspcas9(data_dir)
    moreno_mateos_df = load_moreno_mateos(data_dir)
    
    # Validate
    logger.info("\n2. Validating datasets...")
    valid1 = validate_dataset(deepspcas9_df, "DeepSpCas9")
    valid2 = validate_dataset(moreno_mateos_df, "Moreno-Mateos")
    
    if not (valid1 and valid2):
        logger.warning("Some validation issues found, but continuing...")
    
    # Save
    logger.info("\n3. Saving datasets...")
    save_dataset(
        deepspcas9_df,
        raw_dir / 'DeepSpCas9.csv',
        'DeepSpCas9 (Kim et al. 2019)'
    )
    save_dataset(
        moreno_mateos_df,
        raw_dir / 'Moreno-Mateos.csv',
        'Moreno-Mateos (2015)'
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"DeepSpCas9: {len(deepspcas9_df)} samples")
    logger.info(f"  - Activity: {deepspcas9_df['activity'].mean():.3f} ± {deepspcas9_df['activity'].std():.3f}")
    logger.info(f"  - GC content: {deepspcas9_df['gc_content'].mean():.1%}")
    logger.info(f"Moreno-Mateos: {len(moreno_mateos_df)} samples (independent test set)")
    logger.info(f"  - Activity: {moreno_mateos_df['activity'].mean():.3f} ± {moreno_mateos_df['activity'].std():.3f}")
    logger.info(f"\nData saved to: {raw_dir}")
    
    return deepspcas9_df, moreno_mateos_df


if __name__ == "__main__":
    main()
