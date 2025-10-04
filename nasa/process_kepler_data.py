"""
NASA Kepler Dataset Processing Script
=====================================

This script processes the NASA Kepler cumulative dataset to:
1. Split the dataset into training (66.67%) and testing (33.33%) data (2:1 ratio)
2. Convert koi_disposition to a single binary format:
   - CONFIRMED and CANDIDATE → 1 (planet/candidate)
   - FALSE POSITIVE → 0 (not planet)

Author: AI Assistant
Date: October 4, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    """
    Load the NASA Kepler dataset from CSV file.
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path, comment='#')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

def convert_to_binary(df):
    """
    Convert koi_disposition to a single binary column.
    """
    print("\nConverting koi_disposition to binary format...")

    df_processed = df.copy()

    # Print original value counts
    print("\nOriginal koi_disposition value counts:")
    print(df['koi_disposition'].value_counts())

    # Create single binary column
    df_processed['is_candidate'] = df_processed['koi_disposition'].map({
        'CONFIRMED': 1,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    })

    # Check for unmapped values
    if df_processed['is_candidate'].isna().any():
        print("Warning: Some koi_disposition values could not be mapped!")
        unmapped = df_processed[df_processed['is_candidate'].isna()]['koi_disposition'].unique()
        print(f"Unmapped values: {unmapped}")

    print("\nBinary is_candidate value counts:")
    print(df_processed['is_candidate'].value_counts())

    return df_processed

def split_dataset(df, test_size=1/3, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    print(f"\nSplitting dataset into training and testing sets...")
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['is_candidate']  # keep class balance
    )

    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")

    # Check class distribution
    print("\nClass distribution in training set (is_candidate):")
    print(train_df['is_candidate'].value_counts(normalize=True))
    print("\nClass distribution in testing set (is_candidate):")
    print(test_df['is_candidate'].value_counts(normalize=True))

    return train_df, test_df

def save_datasets(train_df, test_df, output_dir="."):
    """
    Save the processed training and testing datasets to CSV files.
    """
    print(f"\nSaving processed datasets to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "kepler_train_data.csv")
    test_file = os.path.join(output_dir, "kepler_test_data.csv")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Training data saved to: {train_file}")
    print(f"Testing data saved to: {test_file}")

    # Save summary
    summary_file = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("NASA Kepler Dataset Processing Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total samples: {len(train_df) + len(test_df)}\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Testing samples: {len(test_df)}\n")
        f.write(f"Train/Test ratio: {len(train_df)/len(test_df):.2f}:1\n\n")
        f.write("Binary Target Encoding:\n")
        f.write("- koi_disposition: CONFIRMED/CANDIDATE -> 1, FALSE POSITIVE -> 0\n")

    print(f"Summary saved to: {summary_file}")

def main():
    print("NASA Kepler Dataset Processing Script")
    print("=" * 40)

    input_file = "cumulative_2025.10.04_06.23.11.csv"
    output_dir = "processed_data"

    try:
        df = load_data(input_file)
        df_processed = convert_to_binary(df)
        train_df, test_df = split_dataset(df_processed, test_size=1/3, random_state=42)
        save_datasets(train_df, test_df, output_dir)

        print("\nProcessing completed successfully!")
        print("=" * 40)

    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_file}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
