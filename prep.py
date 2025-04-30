import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_data(file_path='data/processed/merged_financial_data.csv'):
    """
    Load the financial dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Columns in dataset: {', '.join(df.columns)}")
    return df

def initial_exploration(df):
    """
    Perform initial data exploration and return basic statistics
    """
    print("\n=== INITIAL DATA EXPLORATION ===")
    
    # Basic info
    print("\n--- Basic Information ---")
    if 'date' in df.columns:
        print(f"Time span: {df['date'].min()} to {df['date'].max()}")
    
    if 'PERMNO' in df.columns:
        print(f"Number of unique securities (PERMNOs): {df['PERMNO'].nunique()}")
    
    if 'COMNAM' in df.columns:
        print(f"Number of unique companies (COMNAMs): {df['COMNAM'].nunique()}")
    
    # Data types and missing values
    print("\n--- Data Types and Missing Values ---")
    missing_data = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing (%)': round(df.isnull().sum() / len(df) * 100, 2)
    })
    print(missing_data.sort_values('Missing (%)', ascending=False))
    
    return missing_data

def clean_data(df):
    """
    Clean the dataset by handling missing values, outliers, and data type conversions
    """
    print("\n=== DATA CLEANING ===")
    df_clean = df.copy()
    
    # Convert date to datetime if it exists
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Create year and month columns for easier analysis
        df_clean['year'] = df_clean['date'].dt.year
        df_clean['month'] = df_clean['date'].dt.month
    
    # Identify columns that exist in the dataset
    financial_cols = [col for col in ['bm', 'pe_op_basic', 'roe', 'roa', 'debt_assets', 
                     'quick_ratio', 'curr_ratio', 'debt_ebitda', 'profit_lct', 
                     'cash_ratio', 'at_turn', 'ptb', 'divyield'] if col in df_clean.columns]
    
    # Convert percentage string columns to float
    for col in financial_cols:
        # Check if the column has string values with percentage symbols
        if df_clean[col].dtype == 'object':
            try:
                # Try to convert string percentages to float
                df_clean[col] = df_clean[col].replace(['%'], '', regex=True).astype(float) / 100
            except:
                # If conversion fails, keep column as is
                pass
    
    return_cols = [col for col in ['RET', 'RETX'] if col in df_clean.columns]
    price_cols = [col for col in ['PRC', 'market_cap', 'log_market_cap'] if col in df_clean.columns]
    
    # Remove rows with missing return or price data if these columns exist
    if return_cols and price_cols:
        before_len = len(df_clean)
        df_clean = df_clean.dropna(subset=return_cols + price_cols)
        print(f"Removed {before_len - len(df_clean)} rows with missing return or price data")
    
    # Forward fill financial metrics if date and PERMNO exist
    if 'date' in df_clean.columns and 'PERMNO' in df_clean.columns and financial_cols:
        df_clean_sorted = df_clean.sort_values(['PERMNO', 'date'])
        for col in financial_cols:
            df_clean[col] = df_clean_sorted.groupby('PERMNO')[col].transform(lambda x: x.fillna(method='ffill'))
    
    # Fill remaining missing values in financial metrics
    for col in financial_cols:
        # Check if SICCD exists and use it for industry grouping
        if 'SICCD' in df_clean.columns:
            # Group by industry (first 2 digits of SIC code)
            df_clean['SIC_2d'] = df_clean['SICCD'].astype(str).str[:2]
            
            # Make sure the column is numeric before calculating median
            if df_clean[col].dtype == 'object':
                # Skip this column if it's not numeric
                print(f"Skipping industry median calculation for column {col} as it contains non-numeric values")
                continue
                
            # Calculate industry median
            medians = df_clean.groupby('SIC_2d')[col].transform('median')
            
            # Fill missing with industry median
            missing_mask = df_clean[col].isna()
            df_clean.loc[missing_mask, col] = medians[missing_mask]
        
        # Any remaining missing values get global median
        if df_clean[col].isna().sum() > 0:
            # Make sure the column is numeric before calculating median
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                print(f"Skipping global median calculation for column {col} as it contains non-numeric values")
    
    # Handle rolling/trailing measures - drop rows with missing rolling measures
    rolling_cols = [col for col in ['rolling_vol', 'mom_12m', 'reversal_1m'] if col in df_clean.columns]
    if rolling_cols:
        before_len = len(df_clean)
        df_clean = df_clean.dropna(subset=rolling_cols)
        print(f"Removed {before_len - len(df_clean)} rows with missing rolling measures")
    
    # Clean outliers using winsorization for financial ratios and metrics
    winsor_cols = [col for col in financial_cols + return_cols + ['volume_millions'] if col in df_clean.columns]
    for col in winsor_cols:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue
            
        lower_bound, upper_bound = df_clean[col].quantile([0.005, 0.995])
        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    print(f"Data cleaned: {df_clean.shape[0]} rows remaining")
    return df_clean

def engineer_features(df):
    """
    Engineer additional features for analysis and modeling
    """
    print("\n=== FEATURE ENGINEERING ===")
    df_eng = df.copy()
    
    # Check if necessary columns exist before feature engineering
    has_returns = 'RET' in df_eng.columns
    has_price = 'PRC' in df_eng.columns
    has_permno = 'PERMNO' in df_eng.columns
    has_date = 'date' in df_eng.columns
    has_volume = 'volume_millions' in df_eng.columns
    
    # Only proceed with time-series feature engineering if we have the necessary columns
    if has_returns and has_permno and has_date:
        # Sort data for time series operations
        df_eng = df_eng.sort_values(['PERMNO', 'date']) if has_permno and has_date else df_eng
        
        # Volatility if not already present
        if 'rolling_vol' not in df_eng.columns:
            print("Calculating 20-day rolling volatility...")
            df_eng['rolling_vol'] = df_eng.groupby('PERMNO')['RET'].transform(
                lambda x: x.rolling(window=20, min_periods=15).std()
            )
        
        # Momentum if not already present
        if 'mom_12m' not in df_eng.columns:
            print("Calculating 12-month momentum...")
            df_eng['mom_12m'] = df_eng.groupby('PERMNO')['RET'].transform(
                lambda x: (np.cumprod(1 + x.shift(1).rolling(window=12, min_periods=10).apply(
                    lambda x: np.prod(1 + x) - 1, raw=True
                )))
            )
        
        # One-month price reversal
        if 'reversal_1m' not in df_eng.columns:
            print("Calculating 1-month price reversal...")
            df_eng['reversal_1m'] = df_eng.groupby('PERMNO')['RET'].transform(lambda x: x.shift(1))
    
    # Price-based features
    if has_price and has_permno and has_date:
        print("Calculating price trends...")
        for window in [5, 10, 20, 50]:
            # Calculate moving averages
            df_eng[f'ma_{window}d'] = df_eng.groupby('PERMNO')['PRC'].transform(
                lambda x: x.rolling(window=window, min_periods=window//2).mean()
            )
            
            # Calculate price relative to moving average
            df_eng[f'pct_from_ma_{window}d'] = (df_eng['PRC'] / df_eng[f'ma_{window}d'] - 1) * 100
    
    # Volatility features
    if has_returns and has_permno and has_date:
        print("Calculating volatility ratios...")
        for short_window, long_window in [(5, 20), (10, 50)]:
            # Short-term volatility
            df_eng[f'vol_{short_window}d'] = df_eng.groupby('PERMNO')['RET'].transform(
                lambda x: x.rolling(window=short_window, min_periods=short_window//2).std()
            )
            
            # Long-term volatility
            df_eng[f'vol_{long_window}d'] = df_eng.groupby('PERMNO')['RET'].transform(
                lambda x: x.rolling(window=long_window, min_periods=long_window//2).std()
            )
            
            # Volatility ratio
            df_eng[f'vol_ratio_{short_window}_{long_window}'] = df_eng[f'vol_{short_window}d'] / df_eng[f'vol_{long_window}d']
    
    # Volume features
    if has_volume and has_permno and has_date:
        print("Calculating volume trends...")
        # Normalize volume by its moving average
        df_eng['vol_ma_20d'] = df_eng.groupby('PERMNO')['volume_millions'].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean()
        )
        df_eng['rel_volume'] = df_eng['volume_millions'] / df_eng['vol_ma_20d']
    
    # Size features
    if 'market_cap' in df_eng.columns and has_date:
        print("Creating size categories...")
        # Create deciles of market cap by date
        df_eng['mkt_cap_decile'] = df_eng.groupby('date')['market_cap'].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
        )
    
    # Create composite scores if component features exist
    has_bm = 'bm' in df_eng.columns
    has_pe = 'pe_op_basic' in df_eng.columns
    has_roe = 'roe' in df_eng.columns
    has_roa = 'roa' in df_eng.columns
    has_debt = 'debt_assets' in df_eng.columns
    
    if has_date and has_bm and has_pe and 'value_score' not in df_eng.columns:
        print("Creating value score...")
        # Value score (combination of book-to-market and earnings yield)
        df_eng['bm_rank'] = df_eng.groupby('date')['bm'].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
        )
        df_eng['pe_rank'] = df_eng.groupby('date')['pe_op_basic'].transform(
            lambda x: pd.qcut(x.replace([np.inf, -np.inf], np.nan), 10, labels=False, duplicates='drop')
        )
        df_eng['value_score'] = (df_eng['bm_rank'] + (9 - df_eng['pe_rank'])) / 18  # Higher is more value
    
    if has_date and has_roe and has_roa and has_debt and 'quality_score' not in df_eng.columns:
        print("Creating quality score...")
        # Quality score (combination of ROE, ROA, and debt-to-assets)
        df_eng['roe_rank'] = df_eng.groupby('date')['roe'].transform(
            lambda x: pd.qcut(x.replace([np.inf, -np.inf], np.nan), 10, labels=False, duplicates='drop')
        )
        df_eng['roa_rank'] = df_eng.groupby('date')['roa'].transform(
            lambda x: pd.qcut(x.replace([np.inf, -np.inf], np.nan), 10, labels=False, duplicates='drop')
        )
        df_eng['debt_rank'] = df_eng.groupby('date')['debt_assets'].transform(
            lambda x: pd.qcut(x.replace([np.inf, -np.inf], np.nan), 10, labels=False, duplicates='drop')
        )
        df_eng['quality_score'] = (df_eng['roe_rank'] + df_eng['roa_rank'] + (9 - df_eng['debt_rank'])) / 27
    
    # Technical indicators
    if has_price and has_permno and has_date:
        print("Calculating technical indicators...")
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df_eng['rsi_14d'] = df_eng.groupby('PERMNO')['PRC'].transform(
            lambda x: calculate_rsi(x, window=14)
        )
    
    # Drop rows with missing engineered features
    exclude_from_dropna = ['COMNAM', 'TICKER']
    cols_to_check = [col for col in df_eng.columns if col not in exclude_from_dropna]
    
    before_len = len(df_eng)
    df_eng = df_eng.dropna(subset=cols_to_check)
    print(f"Removed {before_len - len(df_eng)} rows with missing engineered features")
    
    print(f"Features engineered: {df_eng.shape[1] - df.shape[1]} new features added")
    return df_eng

def normalize_features(df, method='robust'):
    """
    Normalize features to prepare for clustering and modeling
    """
    print("\n=== FEATURE NORMALIZATION ===")
    df_norm = df.copy()
    
    # Define columns to exclude from normalization (identifiers, dates, etc.)
    id_cols = ['PERMNO', 'date', 'year_month', 'TICKER', 'COMNAM', 'EXCHCD', 'SICCD', 'year', 'month', 'SIC_2d']
    # Filter to only include columns that actually exist in the dataframe
    exclude_cols = [col for col in id_cols if col in df_norm.columns]
    
    # Select columns to normalize
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    normalize_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not normalize_cols:
        print("No numeric features to normalize!")
        return df_norm, None
    
    # Choose normalization method
    if method == 'standard':
        scaler = StandardScaler()
    else:  # robust scaling is better for financial data with outliers
        scaler = RobustScaler()
    
    # Split data into identifier columns and feature columns
    identifier_cols = [col for col in df_norm.columns if col not in normalize_cols]
    
    # Apply normalization
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df_norm[normalize_cols]),
        columns=normalize_cols,
        index=df_norm.index
    )
    
    # Reattach identifier columns
    for col in identifier_cols:
        df_normalized[col] = df_norm[col]
    
    print(f"Normalized {len(normalize_cols)} features")
    return df_normalized, scaler

def exploratory_analysis(df):
    """
    Perform exploratory data analysis and create visualizations
    """
    print("\n=== EXPLORATORY ANALYSIS ===")
    
    # Create output directory for plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Check which columns are available
    has_returns = 'RET' in df.columns
    has_market_cap = 'market_cap' in df.columns
    has_log_market_cap = 'log_market_cap' in df.columns
    has_volume = 'volume_millions' in df.columns
    has_date = 'date' in df.columns
    has_year = 'year' in df.columns
    
    # 1. Distribution of returns (if available)
    if has_returns:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['RET'].dropna(), kde=True, bins=100)
        plt.title('Distribution of Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.xlim(-0.2, 0.2)  # Focus on the main part of the distribution
        plt.savefig('plots/preliminary_analysis/returns_distribution.png')
        plt.close()
    
    # 2. Return statistics by year (if available)
    if has_returns and (has_year or has_date):
        # If we don't have a year column but have date, create year
        if not has_year and has_date:
            df['year'] = pd.to_datetime(df['date']).dt.year
            has_year = True
        
        if has_year:
            return_stats = df.groupby('year')['RET'].agg(['mean', 'median', 'std']).reset_index()
            print("\n--- Return Statistics by Year ---")
            print(return_stats)
            
            plt.figure(figsize=(12, 6))
            return_stats.plot(x='year', y=['mean', 'median'], kind='bar', ax=plt.gca())
            plt.title('Mean and Median Returns by Year')
            plt.xlabel('Year')
            plt.ylabel('Return')
            plt.savefig('plots/preliminary_analysis/returns_by_year.png')
            plt.close()
    
    # 3. Correlation matrix of key features (use only available features)
    potential_key_features = ['RET', 'market_cap', 'volume_millions', 'bm', 'pe_op_basic', 
                    'roe', 'roa', 'debt_assets', 'rolling_vol', 'mom_12m', 'reversal_1m',
                    'value_score', 'quality_score']
    
    # Filter to only available features
    key_features = [f for f in potential_key_features if f in df.columns]
    
    if len(key_features) > 1:  # Need at least 2 features for correlation
        corr_matrix = df[key_features].corr()
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix of Key Features')
        plt.tight_layout()
        plt.savefig('plots/preliminary_analysis/correlation_matrix.png')
        plt.close()
    
    # 4. Feature distributions for engineered features (if available)
    potential_engineered_features = [
        'rolling_vol', 'mom_12m', 'reversal_1m', 'rel_volume',
        'pct_from_ma_20d', 'vol_ratio_5_20', 'rsi_14d'
    ]
    
    # Filter to only available features
    engineered_features = [f for f in potential_engineered_features if f in df.columns]
    
    if engineered_features:
        fig, axes = plt.subplots(nrows=len(engineered_features), figsize=(12, 4*len(engineered_features)))
        
        if len(engineered_features) == 1:
            axes = [axes]  # Make it iterable when there's only one feature
            
        for i, feature in enumerate(engineered_features):
            sns.histplot(df[feature].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/preliminary_analysis/engineered_features_distribution.png')
        plt.close()
    
    # 5. Size vs Return relationship (if available)
    if has_returns and (has_log_market_cap or has_market_cap):
        size_col = 'log_market_cap' if has_log_market_cap else 'market_cap'
        plt.figure(figsize=(10, 6))
        sample_size = min(5000, len(df))  # Limit to 5000 points for visibility
        sns.scatterplot(x=size_col, y='RET', data=df.sample(sample_size), alpha=0.5)
        plt.title('Size vs Return Relationship')
        plt.xlabel('Log Market Cap' if has_log_market_cap else 'Market Cap')
        plt.ylabel('Return')
        plt.savefig('plots/preliminary_analysis/size_vs_return.png')
        plt.close()
    
    # 6. Value vs Return relationship (if available)
    if has_returns and 'value_score' in df.columns:
        plt.figure(figsize=(10, 6))
        try:
            df['value_decile'] = pd.qcut(df['value_score'], 10, labels=range(1, 11), duplicates='drop')
            sns.boxplot(x='value_decile', y='RET', data=df)
            plt.title('Value Score Decile vs Return')
            plt.xlabel('Value Score Decile (Higher = More Value)')
            plt.ylabel('Return')
            plt.savefig('plots/preliminary_analysis/value_vs_return.png')
            plt.close()
        except Exception as e:
            print(f"Could not create value vs return plot: {e}")
    
    # 7. Quality vs Return relationship (if available)
    if has_returns and 'quality_score' in df.columns:
        plt.figure(figsize=(10, 6))
        try:
            df['quality_decile'] = pd.qcut(df['quality_score'], 10, labels=range(1, 11), duplicates='drop')
            sns.boxplot(x='quality_decile', y='RET', data=df)
            plt.title('Quality Score Decile vs Return')
            plt.xlabel('Quality Score Decile (Higher = Higher Quality)')
            plt.ylabel('Return')
            plt.savefig('plots/preliminary_analysis/quality_vs_return.png')
            plt.close()
        except Exception as e:
            print(f"Could not create quality vs return plot: {e}")
    
    # 8. Volume vs Volatility (if available)
    if 'rel_volume' in df.columns and 'rolling_vol' in df.columns:
        plt.figure(figsize=(10, 6))
        sample_size = min(5000, len(df))  # Limit to 5000 points for visibility
        sns.scatterplot(x='rel_volume', y='rolling_vol', data=df.sample(sample_size), alpha=0.5)
        plt.title('Relative Volume vs Volatility')
        plt.xlabel('Relative Volume (Volume / 20-day MA)')
        plt.ylabel('20-day Rolling Volatility')
        plt.savefig('plots/preliminary_analysis/volume_vs_volatility.png')
        plt.close()
    
    print("\nExploratory analysis completed. Plots saved in 'plots' directory.")
    return

def analyze_for_unsupervised_learning(df):
    """
    Analyze features specifically for the unsupervised learning implementation
    """
    print("\n=== ANALYSIS FOR UNSUPERVISED LEARNING ===")
    
    # Potential features for clustering
    potential_clustering_features = [
        'rolling_vol', 'mom_12m', 'reversal_1m', 'value_score', 'quality_score',
        'rel_volume', 'pct_from_ma_20d', 'vol_ratio_5_20', 'rsi_14d',
        'market_cap', 'bm', 'pe_op_basic', 'debt_assets'
    ]
    
    # Filter to only available features
    clustering_features = [f for f in potential_clustering_features if f in df.columns]
    
    if not clustering_features:
        print("No suitable features for clustering analysis found in the dataset.")
        return None
    
    # Check feature variance
    feature_variance = df[clustering_features].var().sort_values(ascending=False)
    print("\n--- Feature Variance (Higher is Better for Clustering) ---")
    print(feature_variance)
    
    # Calculate feature importance for future return prediction if we have returns and date/PERMNO
    if 'RET' in df.columns and 'date' in df.columns and 'PERMNO' in df.columns:
        print("\n--- Feature Correlations with Next Month's Return ---")
        df_sorted = df.sort_values(['PERMNO', 'date'])
        df_sorted['next_month_return'] = df_sorted.groupby('PERMNO')['RET'].shift(-1)
        
        # Correlation with next month's return
        feature_correlations = df_sorted[[*clustering_features, 'next_month_return']].corr()['next_month_return'].drop('next_month_return')
        print(feature_correlations.sort_values(ascending=False))
        
        # Create a plot of feature correlations with next month's return
        plt.figure(figsize=(12, 8))
        feature_correlations.sort_values().plot(kind='barh')
        plt.title('Feature Correlation with Next Month Return')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig('plots/preliminary_analysis/feature_return_correlation.png')
        plt.close()
    else:
        feature_correlations = None
        print("Could not calculate feature correlations with future returns due to missing RET, date, or PERMNO columns.")
    
    # Create a feature correlation heatmap for clustering features
    plt.figure(figsize=(12, 10))
    clustering_corr = df[clustering_features].corr()
    sns.heatmap(clustering_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Potential Clustering Features')
    plt.tight_layout()
    plt.savefig('plots/preliminary_analysis/clustering_feature_correlation.png')
    plt.close()
    
    print("\nAnalysis for unsupervised learning completed. Key insights saved in 'plots' directory.")
    return feature_correlations

def save_processed_data(df, output_file='data/processed/processed_financial_data.csv'):
    """
    Save the processed dataframe to a CSV file
    """
    print(f"\nSaving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Data saved: {df.shape[0]} rows and {df.shape[1]} columns")
    return

def main():
    """
    Main function to run the entire data processing pipeline
    """
    # Load data
    df = load_data('data/processed/merged_financial_data.csv')
    
    # Initial exploration
    missing_data = initial_exploration(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Engineer features
    df_eng = engineer_features(df_clean)
    
    # Normalize features
    df_norm, scaler = normalize_features(df_eng, method='robust')
    
    # Exploratory analysis
    exploratory_analysis(df_norm)
    
    # Analyze for unsupervised learning
    feature_correlations = analyze_for_unsupervised_learning(df_norm)
    
    # Save processed data
    save_processed_data(df_norm, 'data/processed/processed_financial_data.csv')
    
    print("\nData processing pipeline completed successfully!")
    return df_norm, feature_correlations

if __name__ == "__main__":
    df_processed, feature_correlations = main()