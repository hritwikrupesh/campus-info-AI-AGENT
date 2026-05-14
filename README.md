# Smart Campus AI Assistant

An AI-powered campus information assistant built using a Hybrid Retrieval-Augmented Generation (RAG) pipeline. The system combines semantic vector search and keyword-based retrieval to provide accurate, explainable, and context-aware responses for campus-related queries.

---

## Overview

The Smart Campus AI Assistant is designed to help students, faculty, and visitors quickly access information related to:

* Departments
* Courses
* Admissions
* Placements
* Faculty
* Campus facilities
* Academic information
* Institutional details

The project uses a production-style RAG architecture with:

* Hybrid Retrieval (Chroma + BM25)
* Answer Validation Layer
* Source Citations
* Streaming Responses
* Evaluation Pipeline
* Interactive Streamlit UI

---

## Features

### Hybrid RAG Pipeline

* Combines semantic vector search (ChromaDB) with BM25 keyword retrieval.
* Improves retrieval accuracy for both semantic and exact-match queries.

### Web Scraping Pipeline

* Crawls and extracts information from the official ANITS website.
* Cleans HTML and stores structured JSON documents for downstream processing.

### Vector Database Generation

* Converts campus documents into embeddings using Sentence Transformers.
* Stores embeddings in ChromaDB for fast semantic retrieval.

### Query Expansion & Retrieval Optimization

* Enhances retrieval quality using contextual query expansion.
* Improves handling of user paraphrasing and vocabulary mismatch.

### Answer Validation Layer

* Uses a second-pass LLM refinement step to improve response completeness and clarity.

### Source Citations

* Displays source URLs alongside responses for explainability and transparency.

### Real-Time Streaming UI

* ChatGPT-style streaming responses using Streamlit.
* Interactive and user-friendly interface.

### Evaluation Pipeline

* Automated evaluation framework for validating retrieval quality and response accuracy.
* Generates accuracy reports and identifies weak retrieval areas.

---

## System Architecture

```text
User Query
     ↓
Streamlit Frontend
     ↓
FastAPI Backend
     ↓
Hybrid Retrieval Pipeline
(BM25 + ChromaDB)
     ↓
Relevant Context Retrieval
     ↓
Groq LLM Generation
     ↓
Answer Validation Layer
     ↓
Final Response + Sources
```

---

## Tech Stack

### Backend

* Python
* FastAPI
* LangChain
* ChromaDB
* BM25
* Groq API
* Sentence Transformers

### Frontend

* Streamlit

### Data Processing

* BeautifulSoup
* Requests
* JSON

### Embedding Models

* all-MiniLM-L6-v2

### LLM

* llama-3.1-8b-instant (Groq)

---

## Project Structure

```text
campus-info-AI-AGENT/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── rag/
│   │   ├── qa_chain.py
│   │   ├── build_vector_db.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── document_loader.py
│   │
│   ├── scraper/
│   │   └── web_scraper.py
│   │
│   └── utils/
│
├── frontend/
│   └── chat_app.py
│
├── data/
│   ├── raw/
│   └── pdf files
│
├── vector_db/
├── requirements.txt
├── README.md
└── .env
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd campus-info-AI-AGENT
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / MacOS

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Running the Project

### Step 1 — Run FastAPI Backend

```bash
uvicorn backend.api.main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

### Step 2 — Run Streamlit Frontend

Open another terminal:

```bash
streamlit run frontend/chat_app.py
```

Frontend runs at:

```text
http://localhost:8501
```

---

## Web Scraping Pipeline

To scrape campus website data:

```bash
python backend/scraper/web_scraper.py
```

This generates cleaned JSON files inside:

```text
data/raw/
```

---

## Building the Vector Database

After scraping:

```bash
python backend/rag/build_vector_db.py
```

This:

* Loads documents
* Splits text into chunks
* Generates embeddings
* Stores vectors in ChromaDB

---

## Example Queries

* What courses are offered at ANITS?
* Tell me about placements.
* What departments are available?
* Explain the admission process.
* What facilities are available on campus?
* Who is the principal?

---

## Evaluation System

The project includes a custom evaluation framework that:

* Generates queries from dataset pages
* Evaluates retrieval quality
* Measures response accuracy
* Detects weak retrieval areas
* Produces evaluation reports

Example metrics:

* Accuracy Percentage
* Failed Query Analysis
* Weak Page Detection
* Retrieval Coverage

---

## Key Highlights

* Hybrid Search Architecture
* Explainable AI Responses
* Real-Time Streaming UI
* Context-Aware Retrieval
* Automated Evaluation Pipeline
* Production-Style RAG Workflow

---

## Future Improvements

* Chat History Persistence
* Multi-turn Conversation Memory
* Feedback-based Learning
* Role-based Access
* Voice Assistant Integration
* Multi-language Support

---

## Resume Project Description

### Smart Campus AI Assistant — Python, LangChain, ChromaDB, BM25, Groq, Streamlit

* Built an AI-powered campus assistant using a hybrid RAG pipeline combining vector search (Chroma) and BM25 for accurate information retrieval.
* Designed a scalable retrieval and response system to generate structured, context-aware answers from campus data.
* Developed an evaluation system and interactive Streamlit UI with source citations and real-time streaming for explainable responses.

---

## Author

Hritwik Rupesh Gollu

B.Tech — Artificial Intelligence & Machine Learning

GitHub: https://github.com/hritwikrupesh

LinkedIn: https://www.linkedin.com/in/hritwikrupeshgollu/

Deployment Link: https://campus-info-ai-agent-ivqctbnucxztvysrftgpub.streamlit.app/
