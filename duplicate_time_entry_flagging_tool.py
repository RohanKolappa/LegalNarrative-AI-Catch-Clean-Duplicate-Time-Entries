# --- Step 1: Import Required Libraries ---
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Step 2: Load Data ---
def load_data(filepath: str, sheet_name: str):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df.dropna(subset=['Narrative'], inplace=True)
    return df

# --- Step 3: Generate Embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(df):
    df['embedding'] = df['Narrative'].apply(lambda x: model.encode(x, convert_to_tensor=False))
    return df

# --- Step 4: Similarity Checking ---
def compute_similarity(new_vec, historical_vecs):
    if len(historical_vecs) == 0:
        return 0, None  # <- FIXED: always return 2 values
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
            'Date': row['Date'],
            'Timekeeper': row['Timekeeper Name'],
            'Narrative': row['Narrative'],
            'Flag': flag,
            'Similarity Score': similarity,
            'Matched Narrative': match_entry,
            'Suggestion': suggestion
        })

        historical_vecs.append(curr_vec)
        narratives.append(row['Narrative'])

    return pd.DataFrame(flagged_results)

# --- Step 8: Run All ---
if __name__ == "__main__":
    # Example path - replace with your actual file
    filepath = "CodeX Hackathon 2025 Dataset.xlsx"
    df = load_data(filepath, sheet_name="Client #1")
    df = generate_embeddings(df)
    result_df = process_entries(df)
    result_df.to_csv("flagged_time_entries.csv", index=False)
    print("Processing complete. Results saved to 'flagged_time_entries.csv'.")

