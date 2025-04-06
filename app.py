import streamlit as st
import pandas as pd

st.title("LegalNarrative AI – Duplicate Time Entry Flagger")

# Load output file
df = pd.read_csv("flagged_time_entries.csv")
st.write(f"## Total Entries Processed: {len(df)}")
st.dataframe(df)

for _, row in df.iterrows():
    if row['Flag'] in ['Red', 'Yellow']:
        st.markdown(f"### 🚩 {row['Flag']} Flag for {row['Timekeeper']} on {row['Date']}")
        st.write(f"**Narrative:** {row['Narrative']}")
        st.write(f"**Matched Narrative:** {row['Matched Narrative']}")
        st.write(f"**Suggestion:** {row['Suggestion']}")
        st.markdown("---")

