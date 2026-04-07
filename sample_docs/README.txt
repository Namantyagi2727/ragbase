Sample Documents for RAGBase Demo
===================================

This folder contains three placeholder documents for demonstrating RAGBase features:

  sample_report.txt   — Fictional Q3 financial report (tests numeric QA)
  sample_faq.txt      — Product FAQ document (tests semantic retrieval)
  sample_policy.txt   — Fictional HR leave policy (tests cross-document synthesis)

To use RAGBase with your own documents:
  1. Upload your PDF/DOCX/TXT/CSV/Excel files via the Streamlit sidebar.
  2. Click "Build Index" to ingest and index them.
  3. Start asking questions.

Your documents are stored locally in uploaded_docs/ and the index in ragbase_index/.
Neither directory is committed to git.
