import os
import fitz
import chromadb

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from openai import OpenAI

from dotenv import load_dotenv


# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")


# =========================
# OPENAI CLIENT
# =========================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# =========================
# FASTAPI SETUP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST MODEL
# =========================

class ChatRequest(BaseModel):
    question: str


# =========================
# CHROMA DB SETUP
# =========================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="sws_ai_docs"
)


# =========================
# EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# TEXT SPLITTER
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# =========================
# INGEST DOCUMENTS
# =========================

DATA_FOLDER = "data"

def ingest_documents():

    existing_count = collection.count()

    if existing_count > 0:
        print("Documents already ingested.")
        return

    print("Starting document ingestion...")

    documents = []

    for file in os.listdir(DATA_FOLDER):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(DATA_FOLDER, file)

            print(f"Processing: {file}")

            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):

                page = doc.load_page(page_num)

                text = page.get_text()

                if not text.strip():
                    continue

                chunks = splitter.split_text(text)

                for idx, chunk in enumerate(chunks):

                    documents.append({
                        "id": f"{file}_{page_num}_{idx}",
                        "text": chunk,
                        "metadata": {
                            "source": file,
                            "page": page_num + 1,
                            "chunk": idx
                        }
                    })

    print("Generating embeddings...")

    texts = [d["text"] for d in documents]

    embeddings = embedding_model.encode(
        texts
    ).tolist()

    print("Saving to ChromaDB...")

    collection.add(
        ids=[d["id"] for d in documents],
        documents=texts,
        embeddings=embeddings,
        metadatas=[d["metadata"] for d in documents]
    )

    print("Ingestion complete.")


# =========================
# RUN INGESTION ON STARTUP
# =========================

ingest_documents()


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
You are an internal SWS AI policy assistant.

Answer ONLY using the provided document context.

If the answer is not available in the documents,
respond exactly with:

'I don't have that information in the company documents.'

Keep answers concise and accurate.
"""


# =========================
# CHAT ENDPOINT
# =========================

@app.post("/api/chat")
def chat(req: ChatRequest):

    question = req.question

    # =========================
    # EMBED QUESTION
    # =========================

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    # =========================
    # RETRIEVE DOCUMENTS
    # =========================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    retrieved_docs = results["documents"][0]

    retrieved_metadata = results["metadatas"][0]

    context = "\n\n".join(retrieved_docs)

    # =========================
    # CREATE PROMPT
    # =========================

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    # =========================
    # OPENAI RESPONSE
    # =========================

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    # =========================
    # EXTRACT SOURCES
    # =========================

    sources = list(set([
        meta["source"]
        for meta in retrieved_metadata
    ]))

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


# =========================
# ROOT ROUTE
# =========================

@app.get("/")
def home():

    return {
        "message": "SWS AI RAG Chatbot Running"
    }