# Multiple PDF Chatbot

An AI-powered chatbot that allows users to upload multiple PDF documents and ask questions about their contents.

## Features

- Upload multiple PDFs
- Extract and process document text
- Generate embeddings using Gemini
- Store vectors using ChromaDB
- Retrieve relevant context
- Answer questions using Gemini 2.5 Flash

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- ChromaDB

## Installation

```bash
git clone <repo-url>
cd multiple-pdf-chatbot

pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

Run:

```bash
streamlit run app.py
```

## Project Structure

```text
app.py
requirements.txt
README.md
.gitignore
```