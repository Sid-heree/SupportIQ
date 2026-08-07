import pandas as pd
import spacy
from collections import Counter
import os

def extract_keywords():
    print("Loading English NLP Model (spaCy)...")
    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")
    
    print("Loading data...")
    df = pd.read_csv("data/processed/cleaned_tickets.csv").dropna(subset=['body_clean'])
    
    # We will test this on a sample of 100 tickets to see the magic quickly
    sample_texts = df['body_clean'].head(100).tolist()
    
    print("\n--- Extracting Keywords ---")
    all_keywords = []
    
    for i, text in enumerate(sample_texts[:5]):  # Print the first 5 so we can see the results
        # 1. Tokenization (breaking text into words)
        doc = nlp(text)
        
        ticket_keywords = []
        for token in doc:
            # 2. Filtering using POS (Part of Speech) & Stop Words
            # We only want Nouns (e.g., 'laptop', 'refund') and Proper Nouns (e.g., 'Windows', 'Google')
            # We also ignore standard stop words (like 'the', 'is', 'at') and punctuation
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.is_alpha:
                # 3. Lemmatization (converting words to their root, e.g., 'batteries' -> 'battery')
                ticket_keywords.append(token.lemma_.lower())
                all_keywords.extend(ticket_keywords)
                
        print(f"\nTicket {i+1} Snippet: {text[:80]}...")
        print(f"Extracted Keywords: {list(set(ticket_keywords))}")

    # 4. Global Keyword Frequency
    print("\n--- Top 10 Most Common Keywords Across All 100 Sample Tickets ---")
    keyword_freq = Counter(all_keywords)
    for word, freq in keyword_freq.most_common(10):
        print(f"{word}: {freq} times")

if __name__ == "__main__":
    extract_keywords()