import pandas as pd
import json
import os

def prepare_llm_data():
    print("🚀 Loading cleaned tickets for LLM dataset preparation...")
    df = pd.read_csv("data/processed/cleaned_tickets.csv")
    
    # Filter rows that have both a valid ticket body and resolution answer
    df_valid = df.dropna(subset=['body', 'answer']).copy()
    print(f"Found {len(df_valid)} tickets with valid answers.")

    # Create Alpaca-style instruction format
    instruction_data = []
    
    # We will sample 3,000 top pairs for efficient LLM fine-tuning
    sample_df = df_valid.sample(n=min(3000, len(df_valid)), random_state=42)

    SYSTEM_PROMPT = "You are SupportIQ, an expert customer support AI assistant. Provide a clear, professional, and helpful solution to the customer's issue."

    for _, row in sample_df.iterrows():
        entry = {
            "instruction": SYSTEM_PROMPT,
            "input": f"Subject: {row['subject']}\nTicket Body: {row['body']}",
            "output": str(row['answer'])
        }
        instruction_data.append(entry)

    # Save to JSON format
    os.makedirs("data/processed", exist_ok=True)
    output_file = "data/processed/llm_instruction_data.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(instruction_data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Successfully created {len(instruction_data)} instruction pairs.")
    print(f"📁 Saved dataset to '{output_file}'.")

if __name__ == "__main__":
    prepare_llm_data()