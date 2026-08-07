import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import time

def build_similarity_engine():
    print("Loading cleaned dataset...")
    df = pd.read_csv("data/processed/cleaned_tickets.csv").dropna(subset=['subject_clean', 'body_clean']).reset_index(drop=True)
    
    # Combine subject and body for rich context
    df['text'] = df['subject_clean'].astype(str) + " " + df['body_clean'].astype(str)
    
    # To keep it fast for testing, let's use a sample of 5,000 tickets. 
    # You can remove .head(5000) later to process the whole dataset.
    df_sample = df.head(5000).copy()
    sentences = df_sample['text'].tolist()

    print("Loading Sentence Transformer Model (all-MiniLM-L6-v2)...")
    # This is a fast, highly-rated model for semantic text similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding tickets into dense vectors (this may take a minute)...")
    start_time = time.time()
    embeddings = model.encode(sentences, show_progress_bar=True)
    print(f"Encoded {len(sentences)} tickets in {time.time() - start_time:.2f} seconds.")

    print("\nSaving Embeddings and Model for future querying...")
    os.makedirs("embeddings", exist_ok=True)
    joblib.dump(embeddings, "embeddings/ticket_embeddings.pkl")
    df_sample.to_csv("embeddings/embedding_reference_data.csv", index=False)
    
    print("Engine successfully built and saved in 'embeddings/'!")
    return model, embeddings, df_sample

def find_similar_tickets(query, model, embeddings, df, top_k=5):
    print(f"\n--- Searching for: '{query}' ---")
    # 1. Convert the new query into an embedding
    query_embedding = model.encode([query])
    
    # 2. Calculate Cosine Similarity against all historical tickets
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # 3. Get the indices of the top_k most similar tickets
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # 4. Display Results
    for i, idx in enumerate(top_indices, 1):
        score = similarities[idx]
        subject = df.iloc[idx]['subject']
        print(f"\nResult {i} (Similarity: {score:.4f})")
        print(f"Subject: {subject}")
        # Print first 100 characters of the body so it doesn't flood the terminal
        print(f"Body snippet: {str(df.iloc[idx]['body'])[:100]}...")

if __name__ == "__main__":
    # Build the engine
    st_model, ticket_embeddings, reference_df = build_similarity_engine()
    
    # Test the engine with a sample query
    sample_query = "I forgot my password and cannot log into my account. Please help!"
    find_similar_tickets(sample_query, st_model, ticket_embeddings, reference_df, top_k=5)