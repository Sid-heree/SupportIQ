import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def discover_topics(n_topics=5, n_words=8):
    print("Loading text data for Topic Modeling...")
    df = pd.read_csv("data/processed/cleaned_tickets.csv").dropna(subset=['body_clean'])
    
    # We will use the first 5000 tickets for speed
    texts = df['body_clean'].head(5000).tolist()

    print("Vectorizing text (converting to word counts)...")
    # LDA requires raw word counts, not TF-IDF frequencies. We also remove stop words.
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts) # Document-Term Matrix

    print(f"Applying LDA to discover {n_topics} hidden topics...")
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(dtm)

    print("\n--- Discovered Topics ---")
    feature_names = vectorizer.get_feature_names_out()
    
    # Loop through each topic and print the top words
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        # Get the indices of the top 'n_words'
        top_word_indices = topic.argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        print(", ".join(top_words))

if __name__ == "__main__":
    discover_topics(n_topics=5, n_words=8)