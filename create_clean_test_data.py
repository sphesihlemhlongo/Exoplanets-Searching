#!/usr/bin/env python3
"""
Script to create a clean test dataset by removing target and disposition columns.
This creates a realistic test scenario where we don't have the answers.
"""

import pandas as pd
import os

def create_clean_test_data():
    """
    Remove target variables and disposition columns from test data
    """
    # File paths
    input_file = "nasa/processed_data/kepler_test_data.csv"
    output_file = "nasa/processed_data/kepler_test_data_clean.csv"
    
    print("Creating clean test dataset...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file {input_file} not found!")
        return
    
    # Load the test data
    df_test = pd.read_csv(input_file)
    print(f"✅ Loaded test data with shape: {df_test.shape}")
    
    # Columns to remove (target variables and disposition columns)
    columns_to_remove = [
        'is_candidate',           # Target variable (binary)
        'koi_disposition',        # Original disposition (CONFIRMED/CANDIDATE/FALSE POSITIVE)
        'koi_pdisposition'        # Pipeline disposition
    ]
    
    # Check which columns exist in the dataset
    existing_columns_to_remove = [col for col in columns_to_remove if col in df_test.columns]
    missing_columns = [col for col in columns_to_remove if col not in df_test.columns]
    
    print(f"\nColumns to remove: {columns_to_remove}")
    print(f"Columns found in dataset: {existing_columns_to_remove}")
    if missing_columns:
        print(f"Columns not found (will skip): {missing_columns}")
    
    # Remove the columns
    df_test_clean = df_test.drop(columns=existing_columns_to_remove)
    
    print(f"\n✅ Removed {len(existing_columns_to_remove)} columns")
    print(f"Original shape: {df_test.shape}")
    print(f"Clean shape: {df_test_clean.shape}")
    
    # Save the clean dataset
    df_test_clean.to_csv(output_file, index=False)
    print(f"✅ Clean test dataset saved to: {output_file}")
    
    # Show remaining columns
    print(f"\nRemaining columns ({len(df_test_clean.columns)}):")
    for i, col in enumerate(df_test_clean.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Show first few rows
    print(f"\nFirst 3 rows of clean dataset:")
    print(df_test_clean.head(3))
    
    return df_test_clean

if __name__ == "__main__":
    clean_data = create_clean_test_data()
    print("\n🎯 Clean test dataset created successfully!")
    print("   This dataset can now be used for realistic model evaluation")
    print("   where the true labels are unknown (as in real-world scenarios).")