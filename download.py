import wrds
import pandas as pd
import os

# Connect to WRDS
db = wrds.Connection()

# Ensure output directory exists
output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)

# 1. CRSP Monthly Stock File
print("Downloading CRSP Monthly Stock data...")
crsp_query = """
    SELECT *
    FROM crsp.msf
    WHERE date BETWEEN '2000-01-01' AND '2024-12-31'
"""
crsp_df = db.raw_sql(crsp_query)
crsp_df.to_csv(os.path.join(output_dir, "crsp.csv"), index=False)
print("CRSP data saved to crsp.csv")

# 2. WRDS Financial Ratios - Firm Level
print("Downloading WRDS Financial Ratios...")
ratios_query = """
    SELECT *
    FROM wrdsapps.finratio
    WHERE datadate BETWEEN '2000-01-01' AND '2024-12-31'
"""
ratios_df = db.raw_sql(ratios_query)
ratios_df.to_csv(os.path.join(output_dir, "ratios.csv"), index=False)
print("Financial ratios data saved to ratios.csv")

# Close connection
db.close()