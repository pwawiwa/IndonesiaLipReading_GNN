#!/usr/bin/env python3
"""
Compute features for a given partition and feature level.
"""
import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from preprocessing.feature_engineering import process_split_features
from utils import setup_logger, ensure_dir


def main():
    parser = argparse.ArgumentParser(description='Compute features for a partition')
    parser.add_argument('--partition', type=str, required=True, choices=['lips', 'mouth', 'full'],
                       help='Partition name')
    parser.add_argument('--feature-level', type=str, required=True, choices=['B0', 'B1', 'B2', 'B3', 'B4'],
                       help='Feature level')
    parser.add_argument('--extracted-dir', type=str, required=True,
                       help='Directory with extracted .pt files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for features')
    
    args = parser.parse_args()
    
    # Setup
    extracted_dir = Path(args.extracted_dir)
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / args.feature_level
    ensure_dir(feature_dir)
    
    logger = setup_logger('ComputeFeatures', log_file=str(feature_dir / 'compute_features.log'))
    
    logger.info("=" * 60)
    logger.info("COMPUTING FEATURES")
    logger.info("=" * 60)
    logger.info(f"Partition: {args.partition}")
    logger.info(f"Feature Level: {args.feature_level}")
    logger.info(f"Extracted Dir: {extracted_dir}")
    logger.info(f"Output Dir: {feature_dir}")
    logger.info("=" * 60)
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        input_file = extracted_dir / f"{args.partition}_{split}.pt"
        output_file = feature_dir / f"{args.partition}_{split}.pt"
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            continue
        
        if output_file.exists():
            logger.info(f"Features already exist: {output_file}, skipping...")
            continue
        
        logger.info(f"Processing {split} split...")
        logger.info(f"  Input: {input_file}")
        logger.info(f"  Output: {output_file}")
        
        try:
            process_split_features(
                extracted_data_path=str(input_file),
                feature_level=args.feature_level,
                output_path=str(output_file)
            )
            logger.info(f"✓ Successfully computed features for {split}")
        except Exception as e:
            logger.error(f"✗ Failed to compute features for {split}: {e}")
            raise
    
    logger.info("=" * 60)
    logger.info("FEATURE COMPUTATION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

