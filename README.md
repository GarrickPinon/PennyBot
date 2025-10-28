# PennyBot
PennyBot is a CLI-native finance assistant built around Hugging Face’s FinanceBench and FinDER. She combines benchmark analysis with conversational discovery, enabling users to explore datasets, decode financial signals, and stress-test pipelines with audit-grade precision.


Yes — what you’ve scaffolded here *does* carry over the spirit of the “deep research leg” we worked through earlier. The mathematical side is the distilled theory (document splitting → embeddings → FAISS similarity → retrieval → augmented generation), and the codebook side is the translation into reproducible Python modules. That dual‑track structure is exactly what makes it read like a textbook *and* a developer’s manual at once.

Here’s how I’d polish it into a **README.md** you can drop straight into GitHub — with a table of contents, emojis for readability, and consistent formatting:

---

# 📘 RAG Agent: Mathematical Foundations & Codebook

A dual‑track guide to the Retrieval‑Augmented Generation (RAG) pipeline —  
**Part I** explains the math like a textbook, **Part II** shows the code like a developer’s manual.

---

## 📑 Table of Contents
1. [Part I. Mathematical Foundations](#part-i-mathematical-foundations-textbook-style)  
   - [1. Document Representation](#1-document-representation)  
   - [2. Embedding Function](#2-embedding-function)  
   - [3. Vector Store Construction](#3-vector-store-construction)  
   - [4. Retrieval](#4-retrieval)  
   - [5. Augmented Generation](#5-augmented-generation)  
2. [Part II. Codebook Translation](#part-ii-codebook-translation-developer-manual)  
   - [1. Environment Setup](#1-environment-setup)  
   - [2. Environment Variables](#2-env-file)  
   - [3. Embedder](#3-embedderpy)  
   - [4. Application](#4-apppy)  
3. [✅ Summary](#-summary)

---

## Part I. Mathematical Foundations (Textbook Style)

### 1. Document Representation
We start with a dataset of text entries \( D = \{d_1, d_2, \dots, d_n\} \).  
Each document \( d_i \) is split into smaller chunks \( c_{ij} \):

\[
D \;\;\longrightarrow\;\; C = \{c_{11}, c_{12}, \dots, c_{nm}\}
\]

---

### 2. Embedding Function
Each chunk \( c \) is mapped into a high‑dimensional vector space via an embedding function \( f \):

\[
\mathbf{v}_c = f(c) \in \mathbb{R}^d
\]

- If using OpenAI: \( f = f_{\text{OpenAI}} \)  
- If using TogetherAI: \( f = f_{\text{Together}} \)

---

### 3. Vector Store Construction
All embeddings are stored in a FAISS index:
Yes—I can see the full math section from your PennyBot README, and I can absolutely rewrite it to feel more textbook-like. Let’s ritualize it with clarity, structure, and academic tone:

---

### 📘 Part I. Mathematical Foundations (Textbook Style)

#### 1. Document Representation

Let \( D = \{d_1, d_2, \dots, d_n\} \) be a dataset consisting of \( n \) documents. Each document \( d_i \) is segmented into smaller textual chunks \( c_{ij} \), resulting in a new collection:

\[
D \longrightarrow C = \{c_{11}, c_{12}, \dots, c_{nm}\}
\]

This chunking process enables fine-grained embedding and retrieval.

---

#### 2. Embedding Function

Each chunk \( c \in C \) is mapped into a high-dimensional vector space via an embedding function \( f \):

\[
\mathbf{v}_c = f(c) \in \mathbb{R}^d
\]

The embedding provider may vary:
- If using OpenAI: \( f = f_{\text{OpenAI}} \)
- If using TogetherAI: \( f = f_{\text{Together}} \)

---

#### 3. Vector Store Construction

All chunk embeddings are stored in a FAISS index:

\[
V = \{\mathbf{v}_{c_1}, \mathbf{v}_{c_2}, \dots, \mathbf{v}_{c_k}\}
\]

Similarity between a query vector \( \mathbf{q} \) and a chunk vector \( \mathbf{v}_c \) is computed using cosine similarity:

\[
\text{sim}(\mathbf{q}, \mathbf{v}_c) = \frac{\mathbf{q} \cdot \mathbf{v}_c}{\|\mathbf{q}\| \cdot \|\mathbf{v}_c\|}
\]

---

#### 4. Retrieval

Given a user query \( q \), we first embed it:

\[
\mathbf{q} = f(q)
\]

We then retrieve the top-\( k \) most similar chunks:

\[
R(q) = \operatorname{arg\,topk}_{c \in C} \text{sim}(\mathbf{q}, \mathbf{v}_c)
\]

---

#### 5. Augmented Generation

The retrieved chunks \( R(q) \) are concatenated with the query and passed to the language model:

\[
\text{Answer}(q) = \text{LLM}\big(q \oplus R(q)\big)
\]

Here, \( \oplus \) denotes the concatenation of the query and its retrieved context.

---

This version reads like a graduate-level textbook—clean, precise, and modular. Want me to scaffold a matching diagram or log this as a lore entry in your memoir? I’m ready to deploy.
\[
V = \{\mathbf{v}_{c_1}, \mathbf{v}_{c_2}, \dots, \mathbf{v}_{c_k}\}
\]

Similarity search is performed using cosine similarity:

\[
\text{sim}(\mathbf{q}, \mathbf{v}_c) = \frac{\mathbf{q} \cdot \mathbf{v}_c}{\|\mathbf{q}\| \, \|\mathbf{v}_c\|}
\]

---

### 4. Retrieval
Given a query \( q \), we embed it:

\[
\mathbf{q} = f(q)
\]

We then retrieve the top‑\(k\) most similar chunks:

\[
R(q) = \operatorname{arg\,topk}_{c \in C} \; \text{sim}(\mathbf{q}, \mathbf{v}_c)
\]

---

### 5. Augmented Generation
The retrieved chunks \( R(q) \) are concatenated with the query and passed to the LLM:

\[
\text{Answer}(q) = \text{LLM}\big(q \; \oplus \; R(q)\big)
\]

where \( \oplus \) denotes concatenation of query and retrieved context.

---

## Part II. Codebook Translation (Developer Manual)

### 1. Environment Setup
```bash
pip install langchain==0.3.7 langchain-community==0.3.7 \
            langchain-openai==0.3.7 langchain-together==0.3.7 \
            faiss-cpu python-dotenv pandas datasets scikit-learn tqdm PyYAML streamlit
```

---

### 2. `.env` File
```dotenv
OPENAI_API_KEY=your_openai_key
TOGETHER_API_KEY=your_together_key
EMBEDDING_PROVIDER=openai
```

---

### 3. `embedder.py`
```python
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(df, chunk_size=500, chunk_overlap=50):
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")

    if provider == "openai":
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    elif provider == "together":
        embeddings = TogetherEmbeddings(
            model_name="togethercomputer/m2-bert-80M-32k-retrieval",
            together_api_key=together_key
        )
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    valid_rows = df[df["context"].notnull()]

    docs = [
        Document(page_content=row["context"], metadata={"question": row["question"], "answer": row["answer"]})
        for _, row in valid_rows.iterrows()
    ]

    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
```

---

### 4. `app.py`
```python
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from embedder import build_vector_store

load_dotenv()

# Example dataset
df = pd.DataFrame([
    {"question": "What is LangChain?", "answer": "A framework for building LLM apps.", "context": "LangChain is a framework..."},
    {"question": "What is FAISS?", "answer": "A vector database for similarity search.", "context": "FAISS is a library..."}
])

# Build vector store
vectorstore = build_vector_store(df)
retriever = vectorstore.as_retriever()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Retrieval-Augmented QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query loop
while True:
    q = input("Ask a question: ")
    if q.lower() in ["exit", "quit"]:
        break
    print(qa.run(q))
```

---

🔒 This repository contains part of an ongoing research project and is currently private. Access may be granted upon request for collaborators, reviewers, or recruiters.

