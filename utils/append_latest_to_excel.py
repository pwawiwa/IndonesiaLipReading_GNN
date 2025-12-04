#!/usr/bin/env python3
"""
Append latest training results to existing Excel summary file.
This script finds the most recent run_meta.pt and adds it to the Excel file
without overwriting existing data.
"""
import torch
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Optional
import os
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the extract_run_info and create_summary_sheets functions
from utils.generate_excel_summary import extract_run_info, create_summary_sheets, collect_all_results


def find_latest_run(results_dir: Path) -> Optional[Path]:
    """Find the most recently modified run_meta.pt file."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None
    
    latest_file = None
    latest_mtime = 0
    
    for run_meta_file in results_path.rglob('run_meta.pt'):
        if run_meta_file.is_file():
            mtime = os.path.getmtime(run_meta_file)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = run_meta_file
    
    return latest_file.parent if latest_file else None


def append_to_excel(excel_file: Path, results_dir: Path):
    """Append latest training results to existing Excel file."""
    excel_path = Path(excel_file)
    results_path = Path(results_dir)
    
    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        print("Please run generate_excel_summary.py first to create the file.")
        return
    
    print("=" * 80)
    print("APPENDING LATEST TRAINING RESULTS TO EXCEL")
    print("=" * 80)
    print(f"Excel file: {excel_path}")
    print(f"Results directory: {results_path}")
    print()
    
    # Find latest run
    print("Finding latest training run...")
    latest_run_dir = find_latest_run(results_path)
    if not latest_run_dir:
        print("No run_meta.pt files found!")
        return
    
    print(f"Latest run directory: {latest_run_dir}")
    
    # Extract info from latest run
    print("Extracting run information...")
    latest_info = extract_run_info(latest_run_dir)
    if not latest_info:
        print("Failed to extract run information!")
        return
    
    print(f"  Partition: {latest_info.get('partition', 'unknown')}")
    print(f"  Feature Level: {latest_info.get('feature_level', 'unknown')}")
    print(f"  Model: {latest_info.get('model', 'unknown')}")
    print(f"  Seed: {latest_info.get('seed', 'unknown')}")
    print()
    
    # Read existing Excel file
    print("Reading existing Excel file...")
    try:
        xls = pd.ExcelFile(excel_path)
        existing_sheets = {}
        for sheet_name in xls.sheet_names:
            existing_sheets[sheet_name] = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"  Loaded sheet: {sheet_name} ({len(existing_sheets[sheet_name])} rows)")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    # Check if this run already exists
    if 'All_Results' in existing_sheets:
        existing_results = existing_sheets['All_Results']
        # Check by result_path
        if 'result_path' in existing_results.columns:
            if latest_info['result_path'] in existing_results['result_path'].values:
                print(f"\n⚠️  This run already exists in the Excel file!")
                print(f"   Result path: {latest_info['result_path']}")
                response = input("   Do you want to update it? (y/n): ").strip().lower()
                if response != 'y':
                    print("   Skipping...")
                    return
                # Remove old entry
                existing_results = existing_results[
                    existing_results['result_path'] != latest_info['result_path']
                ]
                existing_sheets['All_Results'] = existing_results
                print("   Removed old entry, will add updated version.")
    
    # Collect all results (existing + new)
    print("\nCollecting all results (existing + new)...")
    all_results = collect_all_results(results_path)
    print(f"Found {len(all_results)} total training runs")
    
    # Create updated sheets
    print("\nCreating updated summary sheets...")
    updated_sheets = create_summary_sheets(all_results)
    
    if not updated_sheets:
        print("No sheets created!")
        return
    
    # Write to Excel (overwrites but preserves all data)
    print(f"\nWriting updated Excel file to {excel_path}...")
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for sheet_name, df_sheet in updated_sheets.items():
            # Remove internal timestamp columns before writing
            df_to_write = df_sheet.copy()
            if 'timestamp_datetime' in df_to_write.columns:
                df_to_write = df_to_write.drop(columns=['timestamp_datetime'])
            if 'file_modified_timestamp' in df_to_write.columns:
                df_to_write = df_to_write.drop(columns=['file_modified_timestamp'])
            
            print(f"  Writing sheet: {sheet_name} ({len(df_to_write)} rows)")
            df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print()
    print("=" * 80)
    print("✅ EXCEL FILE UPDATED")
    print("=" * 80)
    print(f"Added/updated run: {latest_info.get('partition')}/{latest_info.get('feature_level')}/{latest_info.get('model')}/seed_{latest_info.get('seed')}")
    print(f"Total runs in file: {len(updated_sheets.get('All_Results', []))}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Append latest training results to Excel summary')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory path')
    parser.add_argument('--output', type=str, default='results/training_summary.xlsx',
                       help='Output Excel file path')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_file = Path(args.output)
    
    append_to_excel(output_file, results_dir)


if __name__ == '__main__':
    main()

