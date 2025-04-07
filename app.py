import streamlit as st
from pathlib import Path
from duplicate_time_entry_flagging_tool import (
    parse_excel,
    parse_pdf_with_llamaparse,
    generate_embeddings,
    process_entries
)

#uploaded_file = st.file_uploader("Upload an Excel or PDF file", type=["xlsx", "xls", "pdf"])
uploaded_file = st.file_uploader(
    "Upload an Excel or PDF file", 
    type=["xlsx", "xls", "pdf"], 
    key="unique_file_uploader"
)

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = Path(uploaded_file.name).suffix.lower()

    if ext in [".xlsx", ".xls"]:
        df = parse_excel(uploaded_file.name)
    elif ext == ".pdf":
        df = parse_pdf_with_llamaparse(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    df = generate_embeddings(df)
    result_df = process_entries(df)

    st.success("Processing complete!")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "flagged_time_entries.csv", "text/csv")

