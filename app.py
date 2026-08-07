import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch

app = Flask(__name__)

print("⏳ Loading SupportIQ Model and Vector Store...")

# 1. Load Embeddings & FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("rag/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 2. Load Tokenizer & Fine-Tuned Model
model_path = "models/llm_supportiq/final_adapter"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cpu", 
    torch_dtype=torch.float32
)

# 3. Setup Pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
llm = HuggingFacePipeline(pipeline=pipe)

# 4. Setup Prompt
prompt_template = """You are SupportIQ, a professional AI customer support agent. 
Use the following resolved tickets (Context) to help write a clear and accurate solution for the new user issue.

Context (Past Resolved Tickets):
{context}

New User Issue:
{question}

Suggested Resolution:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

print("✅ Generative AI Backend Ready!")

# --- NEW: Load Predictive ML Models (Brain 1) ---
print("⏳ Loading Predictive ML Models...")
try:
    # 1. Load Vectorizers
    priority_vectorizer = joblib.load("models/priority models/tfidf_vectorizer.pkl")
    type_vectorizer = joblib.load("models/type/tfidf_vectorizer.pkl")
    queue_vectorizer = joblib.load("models/queue models/tfidf_vectorizer.pkl")
    
    # 2. Load Classifiers (Logistic Regression)
    priority_clf = joblib.load("models/priority models/priority_model_lr.pkl")
    type_clf = joblib.load("models/type/type_model_lr.pkl")
    queue_clf = joblib.load("models/queue models/queue_model_lr.pkl")

    # 3. Load Encoders (Translators: Math back to Text)
    priority_encoder = joblib.load("data/processed/priority_encoder.pkl")
    type_encoder = joblib.load("data/processed/type_encoder.pkl")
    queue_encoder = joblib.load("data/processed/queue_encoder.pkl")
    
    print("✅ All Predictive Models and Encoders Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load predictive models. Error: {e}")
    priority_vectorizer = type_vectorizer = queue_vectorizer = None
    priority_clf = type_clf = queue_clf = None
    priority_encoder = type_encoder = queue_encoder = None

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/tickets", methods=["GET"])
def get_tickets():
    try:
        df = pd.read_csv("data/processed/cleaned_tickets.csv") 
        tickets_data = df.fillna("").head(50).to_dict(orient="records")
        return jsonify(tickets_data)
    except Exception as e:
        print(f"Error loading tickets: {e}") 
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Retrieve relevant tickets
    docs = retriever.invoke(user_query)
    retrieved_context = [doc.page_content for doc in docs]
    context_str = "\n\n".join(retrieved_context)

    # Generate response
    formatted_prompt = prompt.format(context=context_str, question=user_query)
    response = llm.invoke(formatted_prompt)

    # Clean the output
    final_answer = response.split("Suggested Resolution:")[-1].strip()
    final_answer = final_answer.replace('\\n', '\n').strip(' ,')

    # 🧠 BRAIN 1: PREDICTIVE ML INFERENCE
    predicted_priority, predicted_eta, predicted_dept = "N/A", "N/A", "N/A"
    
    if priority_clf and type_clf and queue_clf and priority_encoder and type_encoder and queue_encoder:
        try:
            # 1. Vectorize the text (convert words to numbers)
            vec_priority = priority_vectorizer.transform([user_query])
            vec_type = type_vectorizer.transform([user_query])
            vec_queue = queue_vectorizer.transform([user_query])
            
            # 2. Predict the category (returns numerical class predictions)
            num_priority = priority_clf.predict(vec_priority)[0]
            num_type = type_clf.predict(vec_type)[0]
            num_queue = queue_clf.predict(vec_queue)[0]
            
            # 3. Translate numbers back to English labels
            predicted_priority = str(priority_encoder.inverse_transform([num_priority])[0])
            predicted_dept = str(type_encoder.inverse_transform([num_type])[0])
            predicted_eta = str(queue_encoder.inverse_transform([num_queue])[0])
            
        except Exception as e:
            print(f"⚠️ Inference Error: {e}")
            predicted_priority = "Inference Error"

    return jsonify({
        "answer": final_answer,
        "context": retrieved_context,
        "priority": predicted_priority,
        "eta": predicted_eta,
        "department": predicted_dept
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)