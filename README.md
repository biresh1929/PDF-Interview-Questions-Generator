# PDF Interview Questions Generator

An end-to-end **RAG-based PDF interview question generator** that analyzes uploaded PDF documents and automatically generates high-quality interview questions and answers.  
The system uses **LLaMA 7B**, **Sentence-Transformers embeddings**, and **Pinecone vector search**, and is deployed on **AWS EC2** with full **Docker + CI/CD automation**.

---

## 🚀 Features

- Upload PDF documents via a web interface
- Extract and semantically analyze PDF content
- Generate interview-style **questions and answers**
- Retrieval-Augmented Generation (RAG) for context-aware outputs
- Export generated Q&A pairs to **CSV**
- Fully containerized and production-deployed on AWS

---

## 🧠 Architecture Overview

1. **PDF Upload**
   - PDFs are uploaded via a FastAPI-based web interface.
2. **Text Extraction & Chunking**
   - PDF content is parsed and split into semantically meaningful chunks.
3. **Embedding & Vector Storage**
   - Sentence-Transformers (`all-MiniLM`) generate embeddings.
   - Embeddings are stored in **Pinecone** for fast semantic retrieval.
4. **Question & Answer Generation**
   - Retrieved context is passed to a **LLaMA 7B** model.
   - The model generates structured interview questions and answers.
5. **Export**
   - Generated results are saved and exported as a **CSV file**.

---

## 🛠️ Tech Stack

### Backend & APIs
- **FastAPI**
- **Python**
- **Jinja2 (templating)**

### LLM & RAG
- **LLaMA 7B**
- **Sentence-Transformers (all-MiniLM)**
- **Pinecone Vector Database**

### Infrastructure & Deployment
- **Docker**
- **AWS EC2**
- **Amazon ECR**
- **GitHub Actions (CI/CD)**

---

## 📂 Project Structure

```text
.
├── .github/workflows/     # CI/CD pipelines
├── src/                   # Core application logic (RAG + LLM pipeline)
├── data/                  # Local development data
├── research/              # Experiments and exploration
├── static/                # Static assets (CSS, outputs, uploads)
├── templates/             # Jinja2 templates
├── app.py                 # FastAPI application entry point
└── README.md
````

---

## ⚙️ Running Locally

```bash
# Clone the repository
git clone https://github.com/biresh1929/PDF-Interview-Questions-Generator.git
cd PDF-Interview-Questions-Generator

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app:app --host 0.0.0.0 --port 8080
```

---

## ☁️ Deployment

* Containerized using **Docker**
* Images pushed to **Amazon ECR**
* Deployed on **AWS EC2**
* Automated build and deployment using **GitHub Actions CI/CD**

---

## 📈 Use Cases

* Interview preparation from technical PDFs
* Automated assessment content generation
* Academic and educational material analysis
* Knowledge extraction from large documents

---

## 🔒 Notes

* Designed for **scalable semantic retrieval**
* Easily extensible to support additional document formats
* Production-ready deployment setup

---

## 📄 License

This project is open-source and available for learning and experimentation.

---
