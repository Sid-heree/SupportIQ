import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def engineer_features(input_path, output_dir):
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Combine Subject and Body for ML
    # As per the architecture: subject + body -> Predictions
    print("Combining subject and body into a single 'text' feature...")
    df['text'] = df['subject_clean'].astype(str) + " " + df['body_clean'].astype(str)
    
    # 2. Encode Target Variables (Priority, Queue, Type)
    # We must convert text labels (e.g., "P1", "P2") into numbers (e.g., 0, 1) for the models
    print("Encoding target variables...")
    targets = ['priority', 'queue', 'type']
    encoders = {}
    
    for target in targets:
        le = LabelEncoder()
        df[f'{target}_encoded'] = le.fit_transform(df[target].astype(str))
        encoders[target] = le
        print(f"  -> Encoded {target}: {le.classes_}")
        
    # 3. Save the engineered dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "engineered_tickets.csv")
    df.to_csv(output_path, index=False)
    print(f"\nEngineered data saved to {output_path}")
    
    # 4. Save encoders for later use (crucial for Phase 7 Backend API)
    for target, le in encoders.items():
        encoder_path = os.path.join(output_dir, f"{target}_encoder.pkl")
        joblib.dump(le, encoder_path)
        print(f"Saved {target} encoder to {encoder_path}")

if __name__ == "__main__":
    # Ensure these paths match your VS Code setup
    INPUT_FILE = "data/processed/cleaned_tickets.csv"
    OUTPUT_DIR = "data/processed/"
    
    engineer_features(INPUT_FILE, OUTPUT_DIR)