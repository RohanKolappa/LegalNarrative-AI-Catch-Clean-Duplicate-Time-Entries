# --- Step 1: Import Required Libraries ---
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
from io import BytesIO

# Load environment variables (Railway handles this automatically)
load_dotenv()

# Optional: LlamaParse modern integration
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader

# --- Step 2: Load Data ---
def parse_excel(file):
    df = pd.read_excel(file)
    df.dropna(subset=['Narrative'], inplace=True)
    return df

def parse_pdf_with_llamaparse(file):
    parser = LlamaParse(result_type="text")
    file_extractor = {".pdf": parser}
    pdf_path = "/tmp/uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(file.read())
    documents = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor).load_data()
    
    data = []
    for doc in documents:
        for para in doc.text.split("\n"):
            if para.strip():
                data.append({
                    "Narrative": para.strip(),
                    "Timekeeper Name": "Unknown",
                    "Date": pd.NaT
                })
    return pd.DataFrame(data)

def load_data(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return parse_excel(uploaded_file)
    elif ext == ".pdf":
        return parse_pdf_with_llamaparse(uploaded_file)
    else:
        raise ValueError("Unsupported file format")

# --- Step 3: Generate Embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(df):
    df['embedding'] = df['Narrative'].apply(lambda x: model.encode(x, convert_to_tensor=False))
    return df

# --- Step 4: Similarity Checking ---
def compute_similarity(new_vec, historical_vecs):
    if len(historical_vecs) == 0:
        return 0, None
    sims = cosine_similarity([new_vec], historical_vecs)[0]
    return np.max(sims), np.argmax(sims)

# --- Step 5: Flag Entries ---
def flag_similarity(similarity_score):
    if similarity_score > 0.95:
        return 'Red'
    elif similarity_score > 0.85:
        return 'Yellow'
    return 'Green'

# --- Step 6: Suggest Rewording (simulated) ---
def suggest_rewording(narrative):
    return f"Consider specifying more detail in: '{narrative}' (e.g., section or specific topic reviewed)."

# --- Step 7: Process and Flag Entries ---
def process_entries(df):
    flagged_results = []
    historical_vecs = []
    narratives = []

    for idx, row in df.iterrows():
        curr_vec = row['embedding']
        similarity, match_idx = compute_similarity(curr_vec, historical_vecs)
        flag = flag_similarity(similarity)

        suggestion = suggest_rewording(row['Narrative']) if flag in ['Red', 'Yellow'] else ""
        match_entry = narratives[match_idx] if flag in ['Red', 'Yellow'] and match_idx is not None else ""

        flagged_results.append({
            'Date': row.get('Date', None),
            'Timekeeper': row.get('Timekeeper Name', "Unknown"),
            'Narrative': row['Narrative'],
            'Flag': flag,
            'Similarity Score': similarity,
            'Matched Narrative': match_entry,
            'Suggestion': suggestion
        })

        historical_vecs.append(curr_vec)
        narratives.append(row['Narrative'])

    return pd.DataFrame(flagged_results)

# --- Step 8: Streamlit UI ---
st.title("🧠 LegalNarrative AI - Time Entry Checker")
uploaded_file = st.file_uploader("Upload an Excel or PDF file", type=["xlsx", "xls", "pdf"])

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        df = generate_embeddings(df)
        result_df = process_entries(df)

        st.success("Processing complete!")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "flagged_time_entries.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

