#!/usr/bin/env python3
"""
TESS Objects of Interest (TOI) Data Processing Script

This script processes the NASA TESS TOI dataset for exoplanet detection machine learning.
It cleans the data, handles missing values, engineers features, and prepares train/test datasets.

Author: Exoplanet Detection Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_toi_data(filepath):
    """
    Load TESS TOI dataset from CSV file
    
    Args:
        filepath (str): Path to the TOI CSV file
        
    Returns:
        pd.DataFrame: Raw TOI dataset
    """
    print("🚀 Loading TESS TOI dataset...")
    
    # Read CSV, handling comments that start with #
    df = pd.read_csv(filepath, comment='#')
    
    print(f"✅ Loaded {len(df)} TOI objects with {len(df.columns)} columns")
    print(f"🎯 TOI range: {df['toi'].min():.2f} to {df['toi'].max():.2f}")
    
    return df

def clean_toi_labels(df):
    """
    Clean TESS disposition labels for binary classification
    
    TESS Dispositions:
    - PC = Planet Candidate
    - CP = Confirmed Planet  
    - KP = Known Planet
    - FP = False Positive
    
    Binary mapping: PC/CP/KP → 1 (Planet), FP → 0 (False Positive)
    
    Args:
        df (pd.DataFrame): Raw TOI dataset
        
    Returns:
        pd.DataFrame: Dataset with cleaned binary labels
    """
    print("\n🏷️  Cleaning TESS disposition labels...")
    
    # Check current label distribution
    if 'tfopwg_disp' in df.columns:
        print("Original disposition distribution:")
        print(df['tfopwg_disp'].value_counts())
        
        # Create binary labels
        # Planet = 1 (PC, CP, KP), False Positive = 0 (FP)
        df['is_planet'] = df['tfopwg_disp'].map({
            'PC': 1,  # Planet Candidate
            'CP': 1,  # Confirmed Planet
            'KP': 1,  # Known Planet
            'FP': 0,  # False Positive
            'APC': 1, # Astrophysical Planet Candidate (treat as planet)
            'FA': 0   # False Alarm (treat as false positive)
        })
        
        # Handle any unmapped values
        unmapped = df[df['is_planet'].isna()]['tfopwg_disp'].unique()
        if len(unmapped) > 0:
            print(f"⚠️  Found unmapped dispositions: {unmapped}")
            # Drop unmapped entries for now
            df = df.dropna(subset=['is_planet'])
        
        print("\nCleaned binary distribution:")
        print(f"Planets (1): {(df['is_planet'] == 1).sum()}")
        print(f"False Positives (0): {(df['is_planet'] == 0).sum()}")
        print(f"Total: {len(df)}")
        
    else:
        print("⚠️  No 'tfopwg_disp' column found!")
        df['is_planet'] = np.nan
    
    return df

def select_toi_features(df):
    """
    Select relevant features for exoplanet detection from TOI dataset
    
    Key Features:
    - Planet parameters: orbital period, radius, transit depth/duration, temperature
    - Stellar parameters: magnitude, distance, temperature, surface gravity, radius
    - Error columns for feature engineering
    
    Args:
        df (pd.DataFrame): TOI dataset with cleaned labels
        
    Returns:
        pd.DataFrame: Dataset with selected features
    """
    print("\n🔍 Selecting relevant TOI features...")
    
    # Core planet features
    planet_features = [
        'pl_orbper',      # Planet Orbital Period [days]
        'pl_trandurh',    # Planet Transit Duration [hours] 
        'pl_trandep',     # Planet Transit Depth [ppm]
        'pl_rade',        # Planet Radius [R_Earth]
        'pl_insol',       # Planet Insolation [Earth flux]
        'pl_eqt',         # Planet Equilibrium Temperature [K]
    ]
    
    # Planet error columns for feature engineering
    planet_errors = [
        'pl_orbpererr1', 'pl_orbpererr2',     # Orbital period errors
        'pl_trandurherr1', 'pl_trandurherr2', # Transit duration errors
        'pl_trandeperr1', 'pl_trandeperr2',   # Transit depth errors
        'pl_radeerr1', 'pl_radeerr2',         # Planet radius errors
        'pl_insolerr1', 'pl_insolerr2',       # Insolation errors
        'pl_eqterr1', 'pl_eqterr2',           # Temperature errors
    ]
    
    # Stellar features
    stellar_features = [
        'st_tmag',        # TESS Magnitude
        'st_dist',        # Stellar Distance [pc]
        'st_teff',        # Stellar Effective Temperature [K]
        'st_logg',        # Stellar log(g) [cm/s^2]
        'st_rad',         # Stellar Radius [R_Sun]
    ]
    
    # Stellar error columns
    stellar_errors = [
        'st_tmagerr1', 'st_tmagerr2',         # TESS magnitude errors
        'st_disterr1', 'st_disterr2',         # Distance errors  
        'st_tefferr1', 'st_tefferr2',         # Temperature errors
        'st_loggerr1', 'st_loggerr2',         # Surface gravity errors
        'st_raderr1', 'st_raderr2',           # Stellar radius errors
    ]
    
    # Coordinate features (might be useful)
    coordinate_features = [
        'ra', 'dec',      # Right Ascension, Declination
        'st_pmra', 'st_pmdec'  # Proper motion
    ]
    
    # Essential columns
    essential_cols = ['toi', 'tid', 'tfopwg_disp', 'is_planet']
    
    # Combine all feature lists
    all_features = (essential_cols + planet_features + planet_errors + 
                   stellar_features + stellar_errors + coordinate_features)
    
    # Select only columns that exist in the dataset
    available_features = [col for col in all_features if col in df.columns]
    missing_features = [col for col in all_features if col not in df.columns]
    
    print(f"✅ Selected {len(available_features)} available features")
    if missing_features:
        print(f"⚠️  Missing features: {missing_features[:10]}...")  # Show first 10
    
    df_selected = df[available_features].copy()
    
    return df_selected

def handle_missing_values(df):
    """
    Handle missing values in TOI dataset
    
    Strategy:
    1. Drop columns with >50% missing values
    2. For numeric columns: impute with median
    3. Report missing value statistics
    
    Args:
        df (pd.DataFrame): TOI dataset with selected features
        
    Returns:
        pd.DataFrame: Dataset with handled missing values
    """
    print("\n🔧 Handling missing values...")
    
    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct.sort_values(ascending=False)
    
    print("Missing value percentages (top 10):")
    print(missing_pct.head(10))
    
    # Identify columns to drop (>50% missing)
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    essential_cols = ['toi', 'tid', 'tfopwg_disp', 'is_planet']
    cols_to_drop = [col for col in cols_to_drop if col not in essential_cols]
    
    if cols_to_drop:
        print(f"\n🗑️  Dropping {len(cols_to_drop)} columns with >50% missing values:")
        print(cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    
    # Impute remaining missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_essential_numeric = [col for col in numeric_cols if col not in essential_cols]
    
    print(f"\n🔢 Imputing missing values in {len(non_essential_numeric)} numeric columns...")
    
    for col in non_essential_numeric:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   Imputed {col} with median: {median_val:.4f}")
    
    # Final missing value check
    remaining_missing = df.isnull().sum().sum()
    print(f"\n✅ Remaining missing values: {remaining_missing}")
    
    return df

def engineer_toi_features(df):
    """
    Engineer additional features from TOI dataset
    
    Features to create:
    - Relative errors (error/value ratios)
    - Transit signal-to-noise ratios
    - Stellar density estimates
    - Planet/star size ratios
    - Temperature categories
    - Orbital period categories
    
    Args:
        df (pd.DataFrame): Cleaned TOI dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    print("\n⚙️  Engineering TOI features...")
    
    # 1. Relative error features
    error_pairs = [
        ('pl_orbper', 'pl_orbpererr1'),
        ('pl_trandurh', 'pl_trandurherr1'),
        ('pl_trandep', 'pl_trandeperr1'),
        ('pl_rade', 'pl_radeerr1'),
        ('pl_insol', 'pl_insolerr1'),
        ('pl_eqt', 'pl_eqterr1'),
        ('st_tmag', 'st_tmagerr1'),
        ('st_dist', 'st_disterr1'),
        ('st_teff', 'st_tefferr1'),
        ('st_logg', 'st_loggerr1'),
        ('st_rad', 'st_raderr1'),
    ]
    
    print("Creating relative error features...")
    for value_col, error_col in error_pairs:
        if value_col in df.columns and error_col in df.columns:
            new_col = f"{value_col}_rel_err"
            df[new_col] = df[error_col] / (df[value_col] + 1e-10)  # Avoid division by zero
            print(f"   ✓ {new_col}")
    
    # 2. Signal-to-noise ratios
    print("Creating signal-to-noise features...")
    if 'pl_trandep' in df.columns and 'pl_trandeperr1' in df.columns:
        df['pl_trandep_snr'] = df['pl_trandep'] / (df['pl_trandeperr1'] + 1e-10)
        print("   ✓ pl_trandep_snr")
    
    if 'pl_rade' in df.columns and 'pl_radeerr1' in df.columns:
        df['pl_rade_snr'] = df['pl_rade'] / (df['pl_radeerr1'] + 1e-10)
        print("   ✓ pl_rade_snr")
    
    # 3. Planet/star ratios
    print("Creating planet/star ratio features...")
    if 'pl_rade' in df.columns and 'st_rad' in df.columns:
        # Convert stellar radius from solar to Earth radii (1 R_sun = 109.2 R_earth)
        df['planet_star_radius_ratio'] = df['pl_rade'] / (df['st_rad'] * 109.2 + 1e-10)
        print("   ✓ planet_star_radius_ratio")
    
    # 4. Stellar density estimate
    print("Creating stellar density features...")
    if 'st_rad' in df.columns and 'st_logg' in df.columns:
        # Stellar density ∝ g / R^2 (simplified)
        df['st_density_proxy'] = df['st_logg'] / (df['st_rad']**2 + 1e-10)
        print("   ✓ st_density_proxy")
    
    # 5. Temperature categories
    print("Creating temperature categories...")
    if 'pl_eqt' in df.columns:
        df['pl_temp_category'] = pd.cut(df['pl_eqt'], 
                                       bins=[0, 300, 600, 1000, 2000, np.inf],
                                       labels=['cold', 'temperate', 'warm', 'hot', 'scorching'])
        print("   ✓ pl_temp_category")
    
    if 'st_teff' in df.columns:
        df['st_temp_category'] = pd.cut(df['st_teff'],
                                       bins=[0, 3500, 5000, 6500, 10000, np.inf],
                                       labels=['M_dwarf', 'K_dwarf', 'G_dwarf', 'F_dwarf', 'hot_star'])
        print("   ✓ st_temp_category")
    
    # 6. Orbital period categories
    print("Creating orbital period categories...")
    if 'pl_orbper' in df.columns:
        df['pl_period_category'] = pd.cut(df['pl_orbper'],
                                         bins=[0, 1, 10, 100, 1000, np.inf],
                                         labels=['ultra_short', 'short', 'intermediate', 'long', 'very_long'])
        print("   ✓ pl_period_category")
    
    # 7. Planet size categories
    print("Creating planet size categories...")
    if 'pl_rade' in df.columns:
        df['pl_size_category'] = pd.cut(df['pl_rade'],
                                       bins=[0, 1.25, 2.0, 4.0, 11.2, np.inf],
                                       labels=['sub_earth', 'super_earth', 'mini_neptune', 'neptune', 'jupiter'])
        print("   ✓ pl_size_category")
    
    # 8. Insolation categories (habitability proxy)
    print("Creating insolation categories...")
    if 'pl_insol' in df.columns:
        df['pl_habitable_zone'] = pd.cut(df['pl_insol'],
                                        bins=[0, 0.5, 1.5, 10, 1000, np.inf],
                                        labels=['cold_zone', 'habitable_zone', 'warm_zone', 'hot_zone', 'scorched'])
        print("   ✓ pl_habitable_zone")
    
    print(f"\n✅ Feature engineering complete! Dataset now has {len(df.columns)} columns")
    
    return df

def prepare_ml_dataset(df):
    """
    Prepare final dataset for machine learning
    
    Steps:
    1. Separate features and labels
    2. Handle categorical variables
    3. Scale numerical features
    4. Split into train/test sets
    
    Args:
        df (pd.DataFrame): Engineered TOI dataset
        
    Returns:
        dict: Dictionary with X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print("\n🎯 Preparing ML dataset...")
    
    # Remove rows with missing target labels
    df_clean = df.dropna(subset=['is_planet']).copy()
    print(f"Dataset size after removing missing labels: {len(df_clean)}")
    
    # Separate features and target
    feature_cols = [col for col in df_clean.columns 
                   if col not in ['toi', 'tid', 'tfopwg_disp', 'is_planet']]
    
    X = df_clean[feature_cols].copy()
    y = df_clean['is_planet'].copy()
    
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Class distribution: {dict(y.value_counts())}")
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    print(f"\nEncoding {len(categorical_cols)} categorical columns...")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   ✓ {col}")
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train class distribution: {dict(y_train.value_counts())}")
    print(f"Test class distribution: {dict(y_test.value_counts())}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled, 
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X.columns),
        'scaler': scaler,
        'label_encoders': label_encoders,
        'raw_data': df_clean
    }

def save_processed_data(data_dict, output_dir):
    """
    Save processed TOI datasets to files
    
    Args:
        data_dict (dict): Dictionary containing processed datasets
        output_dir (str): Output directory path
    """
    print(f"\n💾 Saving processed data to {output_dir}...")
    
    # Save train and test sets
    train_data = pd.concat([data_dict['X_train'], data_dict['y_train']], axis=1)
    test_data = pd.concat([data_dict['X_test'], data_dict['y_test']], axis=1)
    
    train_path = f"{output_dir}/toi_train_data.csv"
    test_path = f"{output_dir}/toi_test_data.csv"
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"✅ Saved training data: {train_path}")
    print(f"✅ Saved test data: {test_path}")
    
    # Save feature information
    feature_info = {
        'feature_names': data_dict['feature_names'],
        'n_features': len(data_dict['feature_names']),
        'n_train_samples': len(data_dict['X_train']),
        'n_test_samples': len(data_dict['X_test']),
        'train_class_dist': dict(data_dict['y_train'].value_counts()),
        'test_class_dist': dict(data_dict['y_test'].value_counts())
    }
    
    # Create summary report
    summary_text = f"""TESS TOI Dataset Processing Summary
=================================

Dataset Overview:
- Training samples: {feature_info['n_train_samples']}
- Test samples: {feature_info['n_test_samples']}
- Total features: {feature_info['n_features']}

Class Distribution:
- Training: {feature_info['train_class_dist']}
- Test: {feature_info['test_class_dist']}

Key Features:
{chr(10).join([f"- {feat}" for feat in data_dict['feature_names'][:20]])}
{'...' if len(data_dict['feature_names']) > 20 else ''}

Files Generated:
- toi_train_data.csv: Training dataset ({feature_info['n_train_samples']} samples)
- toi_test_data.csv: Test dataset ({feature_info['n_test_samples']} samples)
- toi_dataset_summary.txt: This summary file

The dataset is now ready for exoplanet detection machine learning!
"""
    
    summary_path = f"{output_dir}/toi_dataset_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"✅ Saved summary: {summary_path}")

def main():
    """Main processing pipeline for TESS TOI dataset"""
    
    print("🌟 TESS TOI Dataset Processing Pipeline")
    print("=" * 50)
    
    # Configuration
    input_file = "TOI_2025.10.04_06.24.39.csv"
    output_dir = "toi-processed-data"
    
    try:
        # Step 1: Load data
        df = load_toi_data(input_file)
        
        # Step 2: Clean labels  
        df = clean_toi_labels(df)
        
        # Step 3: Select features
        df = select_toi_features(df)
        
        # Step 4: Handle missing values
        df = handle_missing_values(df)
        
        # Step 5: Engineer features
        df = engineer_toi_features(df)
        
        # Step 6: Prepare ML dataset
        ml_data = prepare_ml_dataset(df)
        
        # Step 7: Save processed data
        save_processed_data(ml_data, output_dir)
        
        print("\n🎉 TOI dataset processing completed successfully!")
        print(f"📊 Ready for machine learning with {len(ml_data['feature_names'])} features")
        print(f"🎯 {len(ml_data['X_train'])} training samples, {len(ml_data['X_test'])} test samples")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()