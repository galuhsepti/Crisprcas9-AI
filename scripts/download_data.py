#!/usr/bin/env python3
"""
Download and prepare CRISPR-Cas9 sgRNA dataset.

This script downloads the Azimuth dataset and prepares it for training.
"""

import sys
import subprocess
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_azimuth():
    """Install azimuth package."""
    logger.info("Installing azimuth package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "azimuth"])
        logger.info("Azimuth installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install azimuth: {e}")
        return False


def find_azimuth_data():
    """Find azimuth package data directory."""
    try:
        import azimuth
        azimuth_dir = Path(azimuth.__file__).parent
        logger.info(f"Found azimuth at: {azimuth_dir}")
        return azimuth_dir
    except ImportError:
        logger.error("Azimuth package not found")
        return None


def load_azimuth_training_data(azimuth_dir):
    """Load training data from azimuth package."""
    logger.info("Loading azimuth training data...")
    
    # Look for data files
    data_files = {
        'model_pickle': azimuth_dir / 'saved_models' / 'V3_model_full.pickle',
        'training_data': azimuth_dir / 'data' / 'all_data.pickle',
    }
    
    for name, path in data_files.items():
        if path.exists():
            logger.info(f"Found {name}: {path}")
    
    # Try to load the model and extract training data
    try:
        import pickle
        # List all files in saved_models
        saved_models_dir = azimuth_dir / 'saved_models'
        if saved_models_dir.exists():
            logger.info(f"Files in saved_models: {list(saved_models_dir.iterdir())}")
        
        # Try to find training data
        data_dir = azimuth_dir / 'data'
        if data_dir.exists():
            logger.info(f"Files in data: {list(data_dir.iterdir())}")
            
            # Look for pickle files
            for pkl_file in data_dir.glob('*.pickle'):
                logger.info(f"Found pickle file: {pkl_file}")
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f"Loaded data type: {type(data)}")
                    if isinstance(data, dict):
                        logger.info(f"Data keys: {data.keys()}")
                    return data
                except Exception as e:
                    logger.warning(f"Could not load {pkl_file}: {e}")
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading azimuth data: {e}")
        return None


def create_sample_dataset(output_dir):
    """Create a sample dataset for testing."""
    logger.info("Creating sample dataset for testing...")
    
    # Sample 30-mer sequences with activity scores
    # Based on Doench 2016 format
    sample_data = {
        'id': [f'sgRNA_{i:03d}' for i in range(1, 51)],
        'sequence': [
            # Valid sequences with varying activity
            'ACAGCTGATCTCCAGATATGACCATGGGTT',
            'CAGCTGATCTCCAGATATGACCATGGGTTT',
            'CCAGAAGTTTGAGCCACAAACCCATGGTCA',
            'GCTGAATCTTCTGTTCTGTCCTTGGGAGA',
            'AGTGAATCTTCTGTTCTGTCCTTGGGAGA',
            'TCTGAATCTTCTGTTCTGTCCTTGGGAGA',
            'ACTGAATCTTCTGTTCTGTCCTTGGGAGA',
            'GCTGAATCTTCTGTTCTGTCCTTGGGTGA',
            'GCTGAATCTTCTGTTCTGTCCTTGGGCGA',
            'GCTGAATCTTCTGTTCTGTCCTTGGGATA',
            'ACAGCTGATCTCCAGATATGACCATGGGTA',
            'ACAGCTGATCTCCAGATATGACCATGGGTC',
            'ACAGCTGATCTCCAGATATGACCATGGGTG',
            'ACAGCTGATCTCCAGATATGACCATGGGCT',
            'ACAGCTGATCTCCAGATATGACCATGGGCC',
            'ACAGCTGATCTCCAGATATGACCATGGGCA',
            'ACAGCTGATCTCCAGATATGACCATGGGCG',
            'ACAGCTGATCTCCAGATATGACCATGGGGA',
            'ACAGCTGATCTCCAGATATGACCATGGGGC',
            'ACAGCTGATCTCCAGATATGACCATGGGGT',
            'ACAGCTGATCTCCAGATATGACCATGGGAA',
            'ACAGCTGATCTCCAGATATGACCATGGGAC',
            'ACAGCTGATCTCCAGATATGACCATGGGAG',
            'ACAGCTGATCTCCAGATATGACCATGGGAT',
            'ACAGCTGATCTCCAGATATGACCATGGGGG',
            'ACAGCTGATCTCCAGATATGACCATGGTTT',
            'ACAGCTGATCTCCAGATATGACCATGGTAA',
            'ACAGCTGATCTCCAGATATGACCATGGTAC',
            'ACAGCTGATCTCCAGATATGACCATGGTAG',
            'ACAGCTGATCTCCAGATATGACCATGGTAT',
            'ACAGCTGATCTCCAGATATGACCATGGTCA',
            'ACAGCTGATCTCCAGATATGACCATGGTCC',
            'ACAGCTGATCTCCAGATATGACCATGGTCG',
            'ACAGCTGATCTCCAGATATGACCATGGTCT',
            'ACAGCTGATCTCCAGATATGACCATGGTGA',
            'ACAGCTGATCTCCAGATATGACCATGGTGC',
            'ACAGCTGATCTCCAGATATGACCATGGTGG',
            'ACAGCTGATCTCCAGATATGACCATGGTGT',
            'ACAGCTGATCTCCAGATATGACCATGGTTA',
            'ACAGCTGATCTCCAGATATGACCATGGTTC',
            'ACAGCTGATCTCCAGATATGACCATGGTTG',
            'ACAGCTGATCTCCAGATATGACCATGGTCT',
            'ACAGCTGATCTCCAGATATGACCATGGATA',
            'ACAGCTGATCTCCAGATATGACCATGGATC',
            'ACAGCTGATCTCCAGATATGACCATGGATG',
            'ACAGCTGATCTCCAGATATGACCATGGATT',
            'ACAGCTGATCTCCAGATATGACCATGGCTA',
            'ACAGCTGATCTCCAGATATGACCATGGCTC',
            'ACAGCTGATCTCCAGATATGACCATGGCTG',
            'ACAGCTGATCTCCAGATATGACCATGGCTT',
        ],
        'activity': np.random.uniform(0.2, 0.9, 50).tolist()
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    output_path = Path(output_dir) / 'sample_dataset.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created sample dataset with {len(df)} sequences")
    logger.info(f"Saved to: {output_path}")
    
    return df


def main():
    """Main function."""
    # Create directories
    data_dir = Path(__file__).parent.parent / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to install and use azimuth
    azimuth_dir = None
    
    if install_azimuth():
        azimuth_dir = find_azimuth_data()
    
    if azimuth_dir:
        # Try to load azimuth data
        data = load_azimuth_training_data(azimuth_dir)
        
        if data is not None:
            logger.info("Successfully loaded azimuth data")
            # Save for later use
            import pickle
            output_path = raw_dir / 'azimuth_data.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved azimuth data to: {output_path}")
        else:
            logger.warning("Could not load azimuth data, creating sample dataset")
            create_sample_dataset(raw_dir)
    else:
        logger.warning("Azimuth not available, creating sample dataset")
        create_sample_dataset(raw_dir)
    
    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main()
