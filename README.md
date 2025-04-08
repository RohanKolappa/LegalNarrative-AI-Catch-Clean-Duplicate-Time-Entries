# LegalNarrative-AI-Catch-Clean-Duplicate-Time-Entries
Project for LLM x Law Hackathon @Stanford #5 (https://law.stanford.edu/event/llm-x-law-hackathon-stanford-5/)

Challenge:
Law Firm clients are using AI to flag and kick back what they suspect are duplicative billing entries, often
incorrectly, creating more work for firm attorneys and staff. The solution is to develop a tool that flags
entries likely to be kicked back based on time entry narrative wording and similarity to other entries.
Yellow flags indicate entries that appear duplicative but are probably correct, while red flags indicate
likely duplicates that should be addressed. The tool could also generate an email to timekeepers with
flagged entries, copying the billing attorney, and suggest wording changes to reduce client flags.
Validation involves providing sanitized entries, noting which ones were flagged by the client, true
duplicates, or incorrect flags, and using this dataset to train a language model on the language that gets
flagged by clients.

Solution:

We're building LegalNarrative AI, a Streamlit-powered web app that helps legal teams automatically detect, flag, and reword repetitive or unclear billing narratives in PDF and Excel files.

Using sentence embeddings and similarity detection, our tool:

Parses .xlsx and .pdf time entries with LlamaParse

Flags similar entries with color codes (🟥 Red, 🟨 Yellow, 🟩 Green)

Suggests rewording for better clarity and accuracy

Outputs a downloadable CSV report to streamline editing and review

📂 Supported File Types: .xlsx, .xls, .pdf

Requires LlamaParse credits and railway/koyeb for streamlit deployment.
Code is currently setup to run locally via Docker.
