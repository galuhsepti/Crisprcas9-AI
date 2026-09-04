#!/usr/bin/env python3
"""
Test feature extraction on the actual CRISPR dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bioinformatics import SequenceFeatureExtractor, extract_one_hot_for_cnn


def main():
    """Main function."""
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Feature Extraction Test")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(raw_dir / 'DeepSpCas9.csv')
    print(f"   Loaded {len(df)} sequences")
    
    # Take a small sample for testing
    sample_size = 100
    df_sample = df.head(sample_size).copy()
    print(f"   Using sample of {sample_size} sequences for testing")
    
    # Initialize feature extractor
    print("\n2. Initializing feature extractor...")
    extractor = SequenceFeatureExtractor(
        context_length=30,
        guide_length=20,
        k_values=[2, 3],
        include_one_hot=False,
        include_gc=True,
        include_composition=True,
        include_kmer=True,
        include_positional=True
    )
    print(f"   Feature count: {extractor.get_feature_count()}")
    print(f"   Feature names (first 10): {extractor.get_feature_names()[:10]}")
    
    # Extract features
    print("\n3. Extracting features...")
    start_time = time.time()
    
    features_df = extractor.extract_features_batch(df_sample['sequence_30mer'].tolist())
    
    elapsed = time.time() - start_time
    print(f"   Extraction time: {elapsed:.2f} seconds")
    print(f"   Features shape: {features_df.shape}")
    print(f"   Features per sequence: {features_df.shape[1]}")
    
    # Display feature statistics
    print("\n4. Feature statistics:")
    print("-" * 40)
    
    # GC content features
    gc_cols = [col for col in features_df.columns if 'gc' in col.lower()]
    print(f"\n   GC content features ({len(gc_cols)}):")
    for col in gc_cols[:5]:
        print(f"     {col}: {features_df[col].mean():.4f} ± {features_df[col].std():.4f}")
    
    # Nucleotide frequency features
    freq_cols = [col for col in features_df.columns if col.startswith('freq_')]
    print(f"\n   Nucleotide frequency features ({len(freq_cols)}):")
    for col in freq_cols:
        print(f"     {col}: {features_df[col].mean():.4f}")
    
    # k-mer features
    kmer_cols = [col for col in features_df.columns if col.startswith('k')]
    print(f"\n   k-mer features ({len(kmer_cols)}):")
    print(f"     Total k-mer features: {len(kmer_cols)}")
    
    # Combine with activity
    print("\n5. Combining features with activity scores...")
    result_df = pd.concat([
        df_sample[['id', 'sequence_30mer', 'guide_sequence', 'activity']].reset_index(drop=True),
        features_df.reset_index(drop=True)
    ], axis=1)
    
    # Save to file
    output_path = processed_dir / 'DeepSpCas9_features_sample.csv'
    result_df.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    
    # Test one-hot encoding for CNN
    print("\n6. Testing one-hot encoding for CNN...")
    one_hot = extract_one_hot_for_cnn(
        df_sample['sequence_30mer'].tolist(),
        context_length=30
    )
    print(f"   One-hot shape: {one_hot.shape}")
    print(f"   Expected shape: ({sample_size}, 30, 4)")
    
    # Save one-hot for later use
    np.save(processed_dir / 'one_hot_sample.npy', one_hot)
    print(f"   Saved one-hot to: {processed_dir / 'one_hot_sample.npy'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Dataset: DeepSpCas9")
    print(f"Sample size: {sample_size}")
    print(f"Features extracted: {features_df.shape[1]}")
    print(f"One-hot encoding: {one_hot.shape}")
    print(f"Output files:")
    print(f"  - {output_path}")
    print(f"  - {processed_dir / 'one_hot_sample.npy'}")
    
    return result_df, one_hot


if __name__ == "__main__":
    main()
