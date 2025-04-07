import streamlit as st
from duplicate_time_entry_flagging_tool import load_data, generate_embeddings, process_entries
from pathlib import Path

st.title("🧠 LegalNarrative AI - Time Entry Checker")

uploaded_file = st.file_uploader("Upload an Excel or PDF file", type=["xlsx", "xls", "pdf"], key="file_upload")

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        df = generate_embeddings(df)
        result_df = process_entries(df)

        st.success("✅ Processing complete!")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "flagged_time_entries.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

