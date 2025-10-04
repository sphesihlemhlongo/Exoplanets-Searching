"""
NASA Kepler Dataset Processing Script
=====================================

This script processes the NASA Kepler cumulative dataset to:
1. Split the dataset into training (66.67%) and testing (33.33%) data (2:1 ratio)
2. Convert koi_disposition and koi_pdisposition columns to binary format:
   - CONFIRMED and CANDIDATE → 1 (candidate)
   - FALSE POSITIVE → 0 (not candidate)

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
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("Loading dataset...")
    # Read CSV file, skipping comment lines that start with '#'
    df = pd.read_csv(file_path, comment='#')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

def convert_to_binary(df):
    """
    Convert disposition columns to binary format.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with binary disposition columns
    """
    print("\nConverting disposition columns to binary format...")
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Print original value counts
    print("\nOriginal koi_disposition value counts:")
    print(df['koi_disposition'].value_counts())
    print("\nOriginal koi_pdisposition value counts:")
    print(df['koi_pdisposition'].value_counts())
    
    # Convert koi_disposition to binary
    # CONFIRMED and CANDIDATE → 1, FALSE POSITIVE → 0
    df_processed['koi_disposition_binary'] = df_processed['koi_disposition'].map({
        'CONFIRMED': 1,
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    })
    
    # Convert koi_pdisposition to binary
    # CANDIDATE → 1, FALSE POSITIVE → 0
    df_processed['koi_pdisposition_binary'] = df_processed['koi_pdisposition'].map({
        'CANDIDATE': 1,
        'FALSE POSITIVE': 0
    })
    
    # Check for any unmapped values
    if df_processed['koi_disposition_binary'].isna().any():
        print("Warning: Some koi_disposition values could not be mapped!")
        unmapped = df_processed[df_processed['koi_disposition_binary'].isna()]['koi_disposition'].unique()
        print(f"Unmapped values: {unmapped}")
    
    if df_processed['koi_pdisposition_binary'].isna().any():
        print("Warning: Some koi_pdisposition values could not be mapped!")
        unmapped = df_processed[df_processed['koi_pdisposition_binary'].isna()]['koi_pdisposition'].unique()
        print(f"Unmapped values: {unmapped}")
    
    # Print binary value counts
    print("\nBinary koi_disposition_binary value counts:")
    print(df_processed['koi_disposition_binary'].value_counts())
    print("\nBinary koi_pdisposition_binary value counts:")
    print(df_processed['koi_pdisposition_binary'].value_counts())
    
    return df_processed

def split_dataset(df, test_size=1/3, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of dataset for testing (1/3 for 1:2 ratio)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) or (train_df, test_df) if no target specified
    """
    print(f"\nSplitting dataset into training and testing sets...")
    print(f"Test size: {test_size:.3f} ({test_size*100:.1f}%)")
    print(f"Train size: {1-test_size:.3f} ({(1-test_size)*100:.1f}%)")
    
    # Split the dataframe into train and test
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['koi_disposition_binary']  # Stratify to maintain class balance
    )
    
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")
    
    # Check class distribution in splits
    print("\nClass distribution in training set (koi_disposition_binary):")
    print(train_df['koi_disposition_binary'].value_counts(normalize=True))
    print("\nClass distribution in testing set (koi_disposition_binary):")
    print(test_df['koi_disposition_binary'].value_counts(normalize=True))
    
    return train_df, test_df

def save_datasets(train_df, test_df, output_dir="."):
    """
    Save the processed training and testing datasets to CSV files.
    
    Args:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Testing dataset
        output_dir (str): Directory to save the files
    """
    print(f"\nSaving processed datasets to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    train_file = os.path.join(output_dir, "kepler_train_data.csv")
    train_df.to_csv(train_file, index=False)
    print(f"Training data saved to: {train_file}")
    
    # Save testing data
    test_file = os.path.join(output_dir, "kepler_test_data.csv")
    test_df.to_csv(test_file, index=False)
    print(f"Testing data saved to: {test_file}")
    
    # Save a summary file
    summary_file = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("NASA Kepler Dataset Processing Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Original dataset shape: {train_df.shape[0] + test_df.shape[0]} rows, {train_df.shape[1]} columns\n")
        f.write(f"Training set shape: {train_df.shape}\n")
        f.write(f"Testing set shape: {test_df.shape}\n")
        f.write(f"Train/Test ratio: {train_df.shape[0]/test_df.shape[0]:.2f}:1\n\n")
        f.write("Binary Encoding:\n")
        f.write("- koi_disposition: CONFIRMED/CANDIDATE -> 1, FALSE POSITIVE -> 0\n")
        f.write("- koi_pdisposition: CANDIDATE -> 1, FALSE POSITIVE -> 0\n\n")
        f.write("Files created:\n")
        f.write(f"- {train_file}\n")
        f.write(f"- {test_file}\n")
        f.write(f"- {summary_file}\n")
    
    print(f"Summary saved to: {summary_file}")

def main():
    """
    Main function to orchestrate the data processing pipeline.
    """
    print("NASA Kepler Dataset Processing Script")
    print("=" * 40)
    
    # Configuration
    input_file = "cumulative_2025.10.04_06.23.11.csv"
    output_dir = "processed_data"
    
    try:
        # Step 1: Load the data
        df = load_data(input_file)
        
        # Step 2: Convert disposition columns to binary
        df_processed = convert_to_binary(df)
        
        # Step 3: Split the dataset (1:2 ratio = test_size of 1/3)
        train_df, test_df = split_dataset(df_processed, test_size=1/3, random_state=42)
        
        # Step 4: Save the processed datasets
        save_datasets(train_df, test_df, output_dir)
        
        print("\n" + "=" * 40)
        print("Dataset processing completed successfully!")
        print("=" * 40)
        
        # Display final statistics
        print(f"\nFinal Statistics:")
        print(f"Total samples: {len(df_processed)}")
        print(f"Training samples: {len(train_df)} ({len(train_df)/len(df_processed)*100:.1f}%)")
        print(f"Testing samples: {len(test_df)} ({len(test_df)/len(df_processed)*100:.1f}%)")
        print(f"Actual Train:Test ratio: {len(train_df)/len(test_df):.2f}:1")
        
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_file}'")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()