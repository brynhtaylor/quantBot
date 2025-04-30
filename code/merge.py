import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load CRSP and financial ratios data"""
    print("Loading datasets...")
    
    # Load CRSP data
    crsp = pd.read_csv('data/raw/crsp.csv')
    # Load financial ratios data
    ratios = pd.read_csv('data/raw/ratios.csv')
    
    print(f"CRSP data shape: {crsp.shape}")
    print(f"Ratios data shape: {ratios.shape}")
    
    return crsp, ratios

def clean_crsp(crsp):
    """Clean CRSP dataset"""
    print("Cleaning CRSP data...")
    
    # Convert date to datetime
    crsp['date'] = pd.to_datetime(crsp['date'])
    
    # Ensure SICCD is string type to avoid conversion issues
    crsp['SICCD'] = crsp['SICCD'].astype(str)
    
    # Filter for common stocks (share codes 10 and 11)
    crsp = crsp[(crsp['SHRCD'] == 10) | (crsp['SHRCD'] == 11)]
    
    # Filter for major exchanges (NYSE, AMEX, NASDAQ)
    crsp = crsp[(crsp['EXCHCD'] == 1) | (crsp['EXCHCD'] == 2) | (crsp['EXCHCD'] == 3)]
    
    # Handle returns
    # Replace missing returns (coded as -99.0, -88.0, etc.) with NaN
    crsp['RET'] = pd.to_numeric(crsp['RET'], errors='coerce')
    crsp['RETX'] = pd.to_numeric(crsp['RETX'], errors='coerce')
    
    # Create market cap
    crsp['PRC'] = crsp['PRC'].abs()  # Price is negative when it's a bid/ask average
    crsp['market_cap'] = crsp['PRC'] * crsp['SHROUT'] / 1000  # SHROUT is in thousands
    
    # Winsorize extreme returns at 1% and 99%
    crsp['RET'] = crsp.groupby(crsp['date'].dt.year)['RET'].transform(
        lambda x: np.clip(x, np.nanpercentile(x, 1), np.nanpercentile(x, 99))
    )
    
    # Log market cap
    crsp['log_market_cap'] = np.log(crsp['market_cap'])
    
    # Monthly volume in millions
    if 'VOL' in crsp.columns:
        crsp['VOL'] = pd.to_numeric(crsp['VOL'], errors='coerce')
        crsp['volume_millions'] = crsp['VOL'] / 1000000
    
    # Create year-month column for merging
    crsp['year_month'] = crsp['date'].dt.year * 100 + crsp['date'].dt.month
    
    # Select relevant columns
    cols_to_keep = ['PERMNO', 'date', 'year_month', 'TICKER', 'COMNAM', 'EXCHCD', 
                   'SICCD', 'PRC', 'RET', 'RETX', 'market_cap', 'log_market_cap']
    
    if 'VOL' in crsp.columns:
        cols_to_keep.append('volume_millions')
    
    crsp = crsp[cols_to_keep]
    
    print(f"Cleaned CRSP data shape: {crsp.shape}")
    return crsp

def clean_ratios(ratios):
    """Clean financial ratios dataset"""
    print("Cleaning financial ratios data...")
    
    # Convert dates to datetime
    for date_col in ['adate', 'qdate', 'public_date']:
        if date_col in ratios.columns:
            ratios[date_col] = pd.to_datetime(ratios[date_col])
    
    # Use public_date as our primary date reference for merging
    ratios['date'] = ratios['public_date']
    
    # Create year-month column for merging
    ratios['year_month'] = ratios['date'].dt.year * 100 + ratios['date'].dt.month
    
    # Handle missing values for key ratios
    key_ratios = ['bm', 'pe_op_basic', 'roe', 'roa', 'debt_assets', 'quick_ratio', 'curr_ratio']
    for ratio in key_ratios:
        if ratio in ratios.columns:
            # Convert to numeric, forcing errors to NaN
            ratios[ratio] = pd.to_numeric(ratios[ratio], errors='coerce')
            
            # Winsorize at 1% and 99% by year to handle outliers
            ratios[ratio] = ratios.groupby(ratios['date'].dt.year)[ratio].transform(
                lambda x: np.clip(x, np.nanpercentile(x, 1), np.nanpercentile(x, 99))
            )
    
    # Select relevant columns (adjust as needed for your analysis)
    cols_to_keep = ['permno', 'gvkey', 'date', 'year_month', 'bm', 'pe_op_basic', 
                   'roe', 'roa', 'debt_assets', 'quick_ratio', 'curr_ratio',
                   'debt_ebitda', 'profit_lct', 'cash_ratio', 'at_turn', 'ptb', 'divyield']
    
    # Keep only columns that exist in the dataset
    cols_to_keep = [col for col in cols_to_keep if col in ratios.columns]
    
    ratios = ratios[cols_to_keep]
    
    # Rename permno to match CRSP convention
    ratios = ratios.rename(columns={'permno': 'PERMNO'})
    
    print(f"Cleaned ratios data shape: {ratios.shape}")
    return ratios

def merge_datasets(crsp, ratios):
    """Merge CRSP and financial ratios datasets"""
    print("Merging datasets...")
    
    # Merge on PERMNO and year_month
    merged = pd.merge(
        crsp,
        ratios,
        on=['PERMNO', 'year_month'],
        how='inner',
        suffixes=('_crsp', '_ratios')
    )
    
    # Use CRSP date as the primary date
    merged = merged.rename(columns={'date_crsp': 'date'})
    
    # Drop duplicate date column from ratios
    if 'date_ratios' in merged.columns:
        merged = merged.drop(columns=['date_ratios'])
    
    print(f"Merged data shape: {merged.shape}")
    return merged

def calculate_factors(merged_data):
    """Calculate common factors for asset pricing"""
    print("Calculating additional factors...")
    
    # Calculate rolling volatility (12-month)
    merged_data['rolling_vol'] = merged_data.groupby('PERMNO')['RET'].transform(
        lambda x: x.rolling(window=12, min_periods=6).std()
    )
    
    # Calculate momentum (12-month)
    merged_data['mom_12m'] = merged_data.groupby('PERMNO')['RET'].transform(
        lambda x: (1 + x.shift(1)).rolling(window=12, min_periods=6).apply(
            lambda y: np.prod(1 + y) - 1, raw=True
        )
    )
    
    # Calculate short-term reversal (1-month lagged return)
    merged_data['reversal_1m'] = merged_data.groupby('PERMNO')['RET'].shift(1)
    
    # Value-Growth composite score (if we have necessary metrics)
    value_cols = [col for col in ['bm', 'pe_op_basic', 'ptb'] if col in merged_data.columns]
    
    if value_cols:
        # Standardize each value metric
        for col in value_cols:
            col_zscore = f"{col}_zscore"
            merged_data[col_zscore] = merged_data.groupby(merged_data['date'].dt.to_period('M'))[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
            )
        
        # Create composite value score (average of z-scores)
        zscore_cols = [f"{col}_zscore" for col in value_cols]
        merged_data['value_score'] = merged_data[zscore_cols].mean(axis=1)
        
        # Drop individual z-score columns
        merged_data = merged_data.drop(columns=zscore_cols)
    
    # Quality composite score
    quality_cols = [col for col in ['roe', 'roa', 'debt_assets', 'quick_ratio'] if col in merged_data.columns]
    
    if quality_cols:
        # Standardize each quality metric
        for col in quality_cols:
            col_zscore = f"{col}_zscore"
            # Flip sign for debt_assets (lower is better)
            mult = -1 if col == 'debt_assets' else 1
            merged_data[col_zscore] = merged_data.groupby(merged_data['date'].dt.to_period('M'))[col].transform(
                lambda x: mult * (x - x.mean()) / x.std() if x.std() != 0 else 0
            )
        
        # Create composite quality score
        zscore_cols = [f"{col}_zscore" for col in quality_cols]
        merged_data['quality_score'] = merged_data[zscore_cols].mean(axis=1)
        
        # Drop individual z-score columns
        merged_data = merged_data.drop(columns=zscore_cols)
    
    return merged_data

def handle_missing_values(data):
    """Handle missing values in the merged dataset"""
    print("Handling missing values...")
    
    # Count missing values before handling
    missing_before = data.isna().sum().sum()
    
    # Fill momentum and volatility measures forward by group
    data['rolling_vol'] = data.groupby('PERMNO')['rolling_vol'].fillna(method='ffill')
    data['mom_12m'] = data.groupby('PERMNO')['mom_12m'].fillna(method='ffill')
    
    # For fundamental ratios, forward-fill by PERMNO (assume ratios stay consistent until updated)
    ratio_cols = [col for col in data.columns if col in [
        'bm', 'pe_op_basic', 'roe', 'roa', 'debt_assets', 'quick_ratio', 
        'curr_ratio', 'debt_ebitda', 'profit_lct', 'cash_ratio', 'at_turn', 'ptb', 'divyield'
    ]]
    
    for col in ratio_cols:
        if col in data.columns:
            data[col] = data.groupby('PERMNO')[col].fillna(method='ffill')
            
            # In case of leading NAs that can't be forward-filled, backward fill
            data[col] = data.groupby('PERMNO')[col].fillna(method='bfill')
    
    # For remaining NAs in factor scores, fill with cross-sectional median by date
    factor_cols = ['value_score', 'quality_score']
    for col in factor_cols:
        if col in data.columns:
            data[col] = data.groupby(data['date'].dt.to_period('M'))[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Count missing values after handling
    missing_after = data.isna().sum().sum()
    print(f"Missing values: {missing_before} before handling, {missing_after} after handling")
    
    # For any remaining missing values, drop rows that have missing RET or market_cap
    data = data.dropna(subset=['RET', 'market_cap'])
    
    print(f"Final data shape after handling missing values: {data.shape}")
    return data

def save_processed_data(data):
    """Save the processed dataset"""
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Ensure all object columns that might cause issues with Parquet are properly handled
    # Convert problematic columns to string type
    object_cols = data.select_dtypes(include=['object']).columns
    for col in object_cols:
        data[col] = data[col].astype(str)
    
    # Save as CSV
    output_path = 'data/processed/merged_financial_data.csv'
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    try:
        # Try to save as Parquet for efficient storage and faster future loading
        parquet_path = 'data/processed/merged_financial_data.parquet'
        data.to_parquet(parquet_path, index=False)
        print(f"Processed data also saved to {parquet_path}")
        return output_path, parquet_path
    except Exception as e:
        print(f"Warning: Could not save to Parquet format: {e}")
        print("Saved as CSV only")
        return output_path, None

def main():
    """Main function to execute the data preparation pipeline"""
    print("Starting financial data preparation pipeline...")
    
    # Step 1: Load raw data
    crsp, ratios = load_data()
    
    # Step 2: Clean individual datasets
    crsp_clean = clean_crsp(crsp)
    ratios_clean = clean_ratios(ratios)
    
    # Step 3: Merge datasets
    merged_data = merge_datasets(crsp_clean, ratios_clean)
    
    # Step 4: Calculate additional factors
    enriched_data = calculate_factors(merged_data)
    
    # Step 5: Handle missing values
    final_data = handle_missing_values(enriched_data)
    
    # Step 6: Save processed data
    output_paths = save_processed_data(final_data)
    
    print("\nData preparation complete!")
    print(f"Final dataset contains {final_data.shape[0]} observations and {final_data.shape[1]} variables")
    print(f"Date range: {final_data['date'].min()} to {final_data['date'].max()}")
    
    # Display sample of unique companies
    company_count = final_data['PERMNO'].nunique()
    print(f"Number of unique companies: {company_count}")
    
    return final_data, output_paths

if __name__ == "__main__":
    main()