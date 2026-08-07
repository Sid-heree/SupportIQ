import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads the dataset from the given filepath."""
    # Corrected f-string: just use the variable name without extra quotes
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_data(df):
    """Performs initial data cleaning."""
    print(f"Original shape: {df.shape}")
    
    # 1. Handle Missing Values
    # If a subject is missing, we can fill it with a placeholder or infer from the body
    df['subject'] = df['subject'].fillna("No Subject")
    
    # Drop rows where the actual answer is missing (since we need it for Phase 5 LLM training)
    df = df.dropna(subset=['answer'])
    
    # Fill missing tags with 'None' or empty strings to avoid breaking multi-label processing later
    tag_columns = [col for col in df.columns if col.startswith('tag_')]
    df[tag_columns] = df[tag_columns].fillna("")

    # 2. Remove Duplicates
    # Drop exact duplicate rows based on the core text content
    duplicates_before = df.duplicated(subset=['subject', 'body', 'language']).sum()
    print(f"Found {duplicates_before} duplicate records. Removing...")
    df = df.drop_duplicates(subset=['subject', 'body', 'language'], keep='first')
    
    # 3. Basic Text Normalization (Lowercase subject and body for consistency)
    df['subject_clean'] = df['subject'].str.lower().str.strip()
    df['body_clean'] = df['body'].str.lower().str.strip()

    print(f"Cleaned shape: {df.shape}")
    return df

def save_data(df, output_path):
    """Saves the cleaned dataset."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # UPDATED PATHS based on your VS Code explorer pane
    # Since you are running from D:\SupportIQ, the paths are relative to that root.
    INPUT_FILE = "data/raw data/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
    OUTPUT_FILE = "data/processed/cleaned_tickets.csv"
    
    # Execute pipeline
    df_raw = load_data(INPUT_FILE)
    df_clean = clean_data(df_raw)
    save_data(df_clean, OUTPUT_FILE)