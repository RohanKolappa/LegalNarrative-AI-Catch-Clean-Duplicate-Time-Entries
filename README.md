# LegalNarrative-AI-Catch-Clean-Duplicate-Time-Entries
📌 Description:  
LegalNarrative AI is a semantic NLP tool that flags law firm time entries that are likely to be rejected by client-side AI systems as duplicates.

It uses embedding models to semantically compare each new entry to historical narratives and classifies them as:
- 🔴 Red: Likely true duplicates that should be revised  
- 🟡 Yellow: Similar entries that are probably legitimate  

🧠 Our tool also suggests rewording to reduce the chance of rejection, helping timekeepers proactively improve narratives.

📬 Email-style summaries notify timekeepers and billing attorneys about flagged entries and suggested improvements.

🗂️ Ingests data from Excel, PDF, and Word documents using LlamaParse (up to 50K+ pages supported).

📊 Performance tracking with Scorecard AI (optional) to monitor precision, false flags, and model tuning over time.

🧑‍💻 Built with:
- Sentence-Transformers (semantic similarity)
- Pandas + Scikit-learn
- Streamlit frontend
- Koyeb for cloud deployment
- LlamaIndex/LlamaParse for scalable ingestion
