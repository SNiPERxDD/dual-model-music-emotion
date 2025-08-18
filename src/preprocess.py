import pandas as pd
import numpy as np

def preprocess_and_check_missing(features_3s_path="features_3_sec.csv", 
                                 melora_path="melora_human_features.csv"):

    # --- Load Data ---
    df_3s = pd.read_csv(features_3s_path)
    df_melora = pd.read_csv(melora_path)
    
    # =============== Check Missing Values ===============
    print("=== Checking Missing Values (features_3_sec.csv) ===")
    missing_3s = df_3s.isnull().sum()
    print(missing_3s[missing_3s > 0])  # Show only columns that have missing values

    print("\n=== Checking Missing Values (melora_human_features.csv) ===")
    missing_melora = df_melora.isnull().sum()
    print(missing_melora[missing_melora > 0])

 

# Run the function if needed
if __name__ == "__main__":
    preprocess_and_check_missing()
