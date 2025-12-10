
---

# 📚 RAG Chatbot with LangChain, ChromaDB & OpenAI

### A Document-Aware Chatbot Powered by Retrieval-Augmented Generation (RAG)

This project is part of my **MLOps & GenAI practical series** where I build real-world, production-ready AI systems.
This repository implements a **Retrieval-Augmented Generation (RAG) chatbot** using:

* **LangChain** (Prompting, Runnables, RAG Pipeline)
* **ChromaDB** (Vector Store for Document Embeddings)
* **OpenAI Embeddings & Chat Models**
* **Streamlit** (UI Dashboard)
* **Docker Compose** (Containerized Deployment)

It allows users to **upload PDF documents**, create embeddings, store them locally, and query them using a chatbot that responds **strictly based on the ingested content**, avoiding hallucinations.

---

# 🚀 Features

### 🔍 **RAG-based Retrieval**

* Document chunks are embedded using **OpenAI text embeddings**
* ChromaDB stores and retrieves relevant sections
* Responses grounded in real context, not hallucinations

### 🤖 **LLM-Powered Chatbot**

* Uses **OpenAI ChatCompletion** (`gpt-3.5-turbo` or better)
* Generates precise answers using LangChain RAG pipeline

### 📂 **Automatic Document Ingestion**

* Upload PDFs into `/data`
* Embeddings automatically generated during ingestion
* Stores them persistently in `/chroma_db`

### 🖥️ **Streamlit UI**

* Clean chatbot interface
* Sidebar options for:

  * Document ingestion
  * Clearing vector store
  * Viewing chat history

### 🐳 **Dockerized Deployment**

* Reproducible environment
* Portable and easy to run anywhere

---

# 🧱 Project Structure

```
rag-chatbot-langchain/
│── chatbot.py               # Main Streamlit app (RAG + Chat)
│── ingest_database.py       # Script to load & embed documents
│── requirements.txt         # Python dependencies
│── Dockerfile               # App image
│── docker-compose.yml       # Service orchestration
│── data/                    # Place your PDF documents here
│── chroma_db/               # Vector store (auto-created)
│── .env                     # Store your OPENAI_API_KEY
└── README.md
```

---

# ⚙️ Tech Stack

### 🧠 Models

* **OpenAIEmbeddings**
* **ChatOpenAI (GPT-3.5 / GPT-4 / GPT-4o)**

### 🔗 Framework

* LangChain (Runnables, Prompts, Document Loaders)

### 🗄️ VectorDB

* ChromaDB (persistent local storage)

### 🖥️ UI

* Streamlit

### 🐳 Deployment

* Docker Compose

---

# 🔑 Environment Variables

Create a `.env` in the project root:

```
OPENAI_API_KEY=your_openai_key_here
```

---

# 🏁 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/rjshk013/mlops-project.git
cd mlops-project/rag-chatbot-langchain
```

---

## 2️⃣ Add Your PDFs

Place your documents inside:

```
/data/
```

Example:

```
data/
 └── Kubernetes-eBook.pdf
```

---

## 3️⃣ Run with Docker Compose (Recommended)

```bash
docker-compose up --build
```

App will be available at:

👉 [http://localhost:8501](http://localhost:8501)

---

# 🗂️ Document Ingestion Flow

1. Click **📥 Ingest Documents** in the sidebar
2. The system loads PDFs → splits text → creates embeddings
3. Stores vectors in **ChromaDB**
4. Ready to query!

---

# 💬 Chat Flow

Once ingestion is complete:

* Type a question in the chat input
* The RAG Chain retrieves the most relevant chunks
* LLM generates an answer based ONLY on document context

Example:

```
“What are the main components of Kubernetes architecture according to the PDF?”
```

---

# 🧪 Testing Your RAG Pipeline

Here are **5 recommended tests**:

### ✔ Test 1 — General Understanding

```
Summarize the document in 5 points.
```

### ✔ Test 2 — Section-Level Query

```
What does the eBook say about Kubernetes control plane components?
```

### ✔ Test 3 — Compare Topics

```
Explain the difference between worker nodes and control plane as described in the PDF.
```

### ✔ Test 4 — Specific Keyword Retrieval

```
What does the document mention about kubelet?
```

### ✔ Test 5 — Out-of-Scope Hallucination Test

```
Explain Kubernetes Istio mesh configuration (if not in the PDF).
```

Expected:
Chatbot should respond:

> “This information is not present in the ingested documents.”

---

# 🛠️ Development Commands (Without Docker)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run ingest script:

```bash
python ingest_database.py
```

Run app locally:

```bash
streamlit run chatbot.py
```

---

# 🌐 Deployment Options

This app can be deployed on:

* AWS EC2
* AWS ECS / Fargate
* Azure WebApps
* GCP Cloud Run
* Docker Swarm
* Kubernetes (Helm chart can be added later)

---

# 🧩 Future Enhancements (Planned)

* 🔄 Multi-file upload from UI
* 📎 Source citation in answers
* 🧠 Document summarization mode
* 🤝 Multi-model routing (OpenAI + local LLMs)
* 🌍 Support for Ollama local inference
* 🔍 Highlight retrieved context inside UI

---

# 🧑‍💻 Author

**Rajesh K**
DevOps | MLOps | Cloud Engineer
GitHub: [https://github.com/rjshk013](https://github.com/rjshk013)

---
