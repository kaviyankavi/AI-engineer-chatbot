SWS AI Policy RAG Chatbot
A Retrieval-Augmented Generation (RAG) chatbot built using FastAPI, ChromaDB, Sentence Transformers, and OpenAI.
This chatbot allows employees to ask natural language questions about company policies and receive accurate, document-grounded answers from internal PDF documents.

Features


PDF document ingestion


Text chunking using LangChain


Embedding generation using Sentence Transformers


Semantic search with ChromaDB


OpenAI-powered grounded responses


FastAPI backend


Simple web-based chatbot UI


Source document tracking


Prevents hallucinated answers



Tech Stack
Backend


Python


FastAPI


ChromaDB


Sentence Transformers


OpenAI API


PyMuPDF


LangChain Text Splitter


Frontend


HTML


CSS


JavaScript



Project Structure
SWS ChatBot/│├── app.py├── index.html├── .env├── chroma_db/├── data/│   ├── policy1.pdf│   ├── policy2.pdf│   └── ...└── README.md

How It Works
1. Document Ingestion


PDF documents are loaded from the data/ folder


Text is extracted using PyMuPDF


Text is split into chunks using RecursiveCharacterTextSplitter


Embeddings are generated using Sentence Transformers


Chunks + embeddings are stored in ChromaDB



2. Retrieval
When a user asks a question:


The question is converted into embeddings


ChromaDB retrieves the most relevant document chunks


Relevant chunks are passed to OpenAI as context



3. Response Generation
The LLM is instructed to:


Answer ONLY from provided document context


Avoid hallucinations


Return fallback response if information is unavailable
