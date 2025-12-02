#!/usr/bin/env python3
"""
Create a config file for training from a template.
"""
import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config, save_config


def main():
    parser = argparse.ArgumentParser(description='Create config from template')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name')
    parser.add_argument('--partition', type=str, required=True,
                       choices=['lips', 'mouth', 'full'],
                       help='Partition name')
    parser.add_argument('--feature-level', type=str, required=True,
                       choices=['B0', 'B1', 'B2', 'B3', 'B4'],
                       help='Feature level')
    parser.add_argument('--template', type=str, required=True,
                       help='Path to template config')
    parser.add_argument('--output', type=str, required=True,
                       help='Output config path')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs (for testing)')
    
    args = parser.parse_args()
    
    # Load template
    template = load_config(args.template)
    
    # Update with partition and feature level
    template['data']['partition'] = args.partition
    template['data']['feature_level'] = args.feature_level
    template['model']['name'] = args.model
    
    # Set paths (use absolute paths)
    project_root = Path(__file__).parent.parent
    template['data']['feature_dir'] = str(project_root / 'features')
    template['training']['output_dir'] = str(project_root / 'results')
    
    # Override epochs if provided (for testing)
    if args.epochs is not None:
        template['training']['epochs'] = args.epochs
        print(f"Overriding epochs to: {args.epochs}", file=sys.stderr)
    
    # Save config
    save_config(template, args.output)
    # Print to stderr so stdout is empty (shell script will use the path directly)
    print(f"Created config: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()

