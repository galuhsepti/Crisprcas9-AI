"""
Preprocessing pipeline for CRISPR-Cas9 sgRNA prediction.

This module provides functions to load, validate, and preprocess
sgRNA datasets for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging

from .validation import (
    validate_sequence,
    normalize_sequence,
    convert_rna_to_dna,
    is_valid_dna
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CRISPRPreprocessor:
    """
    Preprocessing pipeline for CRISPR-Cas9 sgRNA datasets.
    """
    
    def __init__(
        self,
        context_length: int = 30,
        guide_length: int = 20,
        handle_rna: bool = True,
        validate_pam: bool = True,
        remove_duplicates: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize preprocessor.
        
        Args:
            context_length: Expected full context sequence length
            guide_length: Expected guide RNA length
            handle_rna: Convert U to T in sequences
            validate_pam: Validate PAM sequences
            remove_duplicates: Remove duplicate sequences
            random_seed: Random seed for reproducibility
        """
        self.context_length = context_length
        self.guide_length = guide_length
        self.handle_rna = handle_rna
        self.validate_pam = validate_pam
        self.remove_duplicates = remove_duplicates
        self.random_seed = random_seed
        
        # Statistics
        self.stats = {
            'total_sequences': 0,
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'duplicates_removed': 0,
            'rna_converted': 0
        }
    
    def load_dataset(
        self,
        filepath: str,
        sequence_col: str = 'sequence',
        activity_col: str = 'activity',
        id_col: Optional[str] = 'id',
        sep: str = ','
    ) -> pd.DataFrame:
        """
        Load dataset from CSV/TSV file.
        
        Args:
            filepath: Path to data file
            sequence_col: Name of sequence column
            activity_col: Name of activity/score column
            id_col: Name of ID column (optional)
            sep: Delimiter for file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading dataset from {filepath}")
        
        # Load data
        df = pd.read_csv(filepath, sep=sep)
        
        # Standardize column names
        column_mapping = {}
        if sequence_col in df.columns:
            column_mapping[sequence_col] = 'sequence'
        if activity_col in df.columns:
            column_mapping[activity_col] = 'activity'
        if id_col and id_col in df.columns:
            column_mapping[id_col] = 'id'
        
        df = df.rename(columns=column_mapping)
        
        self.stats['total_sequences'] = len(df)
        logger.info(f"Loaded {len(df)} sequences")
        
        return df
    
    def validate_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate all sequences in DataFrame.
        
        Args:
            df: DataFrame with 'sequence' column
            
        Returns:
            DataFrame with validation results added
        """
        logger.info("Validating sequences...")
        
        validation_results = []
        
        for idx, row in df.iterrows():
            sequence = str(row['sequence'])
            
            # Normalize
            sequence = normalize_sequence(sequence)
            
            # Convert RNA if needed
            if self.handle_rna and 'U' in sequence.upper():
                sequence = convert_rna_to_dna(sequence)
                self.stats['rna_converted'] += 1
            
            # Validate
            is_valid, errors = validate_sequence(
                sequence,
                context_length=self.context_length,
                guide_length=self.guide_length,
                check_pam=self.validate_pam
            )
            
            validation_results.append({
                'index': idx,
                'original_sequence': row['sequence'],
                'normalized_sequence': sequence,
                'is_valid': is_valid,
                'errors': errors if errors else None
            })
        
        # Create validation DataFrame
        val_df = pd.DataFrame(validation_results)
        
        # Merge with original
        df = df.copy()
        df['normalized_sequence'] = val_df['normalized_sequence'].values
        df['is_valid'] = val_df['is_valid'].values
        df['validation_errors'] = val_df['errors'].values
        
        # Update stats
        self.stats['valid_sequences'] = df['is_valid'].sum()
        self.stats['invalid_sequences'] = (~df['is_valid']).sum()
        
        logger.info(f"Valid sequences: {self.stats['valid_sequences']}")
        logger.info(f"Invalid sequences: {self.stats['invalid_sequences']}")
        
        return df
    
    def remove_invalid_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid sequences from DataFrame.
        
        Args:
            df: DataFrame with 'is_valid' column
            
        Returns:
            DataFrame with only valid sequences
        """
        initial_count = len(df)
        df = df[df['is_valid'] == True].copy()
        
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} invalid sequences")
        
        return df
    
    def handle_duplicates(
        self,
        df: pd.DataFrame,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Handle duplicate sequences.
        
        Args:
            df: DataFrame with sequences
            keep: Which duplicate to keep ('first', 'last', False)
            
        Returns:
            DataFrame with duplicates handled
        """
        if not self.remove_duplicates:
            return df
        
        initial_count = len(df)
        
        # Check for duplicates by normalized sequence
        if 'normalized_sequence' in df.columns:
            duplicate_mask = df.duplicated(
                subset=['normalized_sequence'],
                keep=keep
            )
        else:
            duplicate_mask = df.duplicated(
                subset=['sequence'],
                keep=keep
            )
        
        df = df[~duplicate_mask].copy()
        
        self.stats['duplicates_removed'] = initial_count - len(df)
        logger.info(f"Removed {self.stats['duplicates_removed']} duplicates")
        
        return df
    
    def normalize_activity(
        self,
        df: pd.DataFrame,
        activity_col: str = 'activity',
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        Normalize activity scores.
        
        Args:
            df: DataFrame with activity column
            activity_col: Name of activity column
            method: Normalization method ('minmax', 'zscore', 'none')
            
        Returns:
            DataFrame with normalized activity
        """
        df = df.copy()
        
        if method == 'none':
            df['normalized_activity'] = df[activity_col]
            return df
        
        # Store raw values
        df['raw_activity'] = df[activity_col]
        
        if method == 'minmax':
            min_val = df[activity_col].min()
            max_val = df[activity_col].max()
            
            if max_val - min_val > 0:
                df['normalized_activity'] = (df[activity_col] - min_val) / (max_val - min_val)
            else:
                df['normalized_activity'] = 0.5
        
        elif method == 'zscore':
            mean_val = df[activity_col].mean()
            std_val = df[activity_col].std()
            
            if std_val > 0:
                df['normalized_activity'] = (df[activity_col] - mean_val) / std_val
            else:
                df['normalized_activity'] = 0.0
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Normalized activity using {method} method")
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            stratify: Whether to stratify by activity
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        np.random.seed(self.random_seed)
        
        # Shuffle indices
        indices = np.random.permutation(len(df))
        
        # Calculate split points
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        
        # Split indices
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Create splits
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        logger.info(f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def extract_guide_sequence(
        self,
        sequence: str,
        context_length: int = 30,
        guide_length: int = 20
    ) -> str:
        """
        Extract guide sequence from context.
        
        Args:
            sequence: Full context sequence (30-mer)
            context_length: Full context length
            guide_length: Guide length to extract
            
        Returns:
            Guide sequence (20bp)
        """
        start = (context_length - guide_length) // 2
        return sequence[start:start + guide_length]
    
    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        sequence_col: str = 'sequence',
        activity_col: str = 'activity',
        normalize: bool = True,
        split: bool = True
    ) -> Dict[str, Any]:
        """
        Run full preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            sequence_col: Name of sequence column
            activity_col: Name of activity column
            normalize: Whether to normalize activity
            split: Whether to split data
            
        Returns:
            Dictionary with preprocessed data and metadata
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Reset stats
        self.stats = {
            'total_sequences': len(df),
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'duplicates_removed': 0,
            'rna_converted': 0
        }
        
        # Step 1: Validate sequences
        df = self.validate_sequences(df)
        
        # Step 2: Remove invalid
        df = self.remove_invalid_sequences(df)
        
        # Step 3: Handle duplicates
        df = self.handle_duplicates(df)
        
        # Step 4: Normalize activity if needed
        if normalize and activity_col in df.columns:
            df = self.normalize_activity(df, activity_col)
        
        # Step 5: Extract guide sequences
        if 'normalized_sequence' in df.columns:
            df['guide_sequence'] = df['normalized_sequence'].apply(
                lambda x: self.extract_guide_sequence(x)
            )
        
        # Step 6: Split if requested
        result = {
            'dataframe': df,
            'stats': self.stats.copy()
        }
        
        if split and len(df) > 0:
            train_df, val_df, test_df = self.split_data(df)
            result['train'] = train_df
            result['validation'] = val_df
            result['test'] = test_df
        
        logger.info("Preprocessing pipeline completed")
        
        return result


def load_azimuth_data(data_dir: str) -> pd.DataFrame:
    """
    Load data from Azimuth package.
    
    Args:
        data_dir: Path to azimuth package directory
        
    Returns:
        DataFrame with sgRNA data
    """
    data_path = Path(data_dir)
    
    # Look for data files
    possible_files = [
        'saved_models/V3_model_full.pickle',
        'data/training_data.csv',
        'data/all_aa_data.csv'
    ]
    
    for file_path in possible_files:
        full_path = data_path / file_path
        if full_path.exists():
            logger.info(f"Found data file: {full_path}")
            # Load based on file type
            if file_path.endswith('.csv'):
                return pd.read_csv(full_path)
            elif file_path.endswith('.pickle'):
                import pickle
                with open(full_path, 'rb') as f:
                    return pickle.load(f)
    
    raise FileNotFoundError(f"No data files found in {data_dir}")


if __name__ == "__main__":
    # Example usage
    preprocessor = CRISPRPreprocessor()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'id': ['sgRNA_1', 'sgRNA_2', 'sgRNA_3'],
        'sequence': [
            'ACAGCTGATCTCCAGATATGACCATGGGTT',  # Valid 30-mer
            'CAGCTGATCTCCAGATATGACCATGGGTTT',  # Valid 30-mer
            'CCAGAAGTTTGAGCCACAAACCCATGGTCA'   # Valid 30-mer
        ],
        'activity': [0.7, 0.8, 0.5]
    })
    
    # Run preprocessing
    result = preprocessor.preprocess_pipeline(sample_data)
    
    print("Preprocessing completed!")
    print(f"Statistics: {result['stats']}")
    print(f"Final dataset shape: {result['dataframe'].shape}")
