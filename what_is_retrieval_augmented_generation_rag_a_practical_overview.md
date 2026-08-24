# What Is Retrieval Augmented Generation (RAG)? A Practical Overview

## Define RAG and its Core Architecture

Retrieval Augmented Generation (RAG) fuses a retrieval engine with a language model. The **retrieval module** queries a vector store or search index, returning a handful of documents that match the user prompt. *Not found in provided sources.* The **generative module** then conditions its decoding on both the prompt and the retrieved passages, producing an answer that blends prior knowledge with fresh evidence. *Not found in provided sources.* RAG enables up‑to‑date answers without retraining the LLM, making it ideal for knowledge‑intensive applications. *Not found in provided sources.* The overall flow is: user query → retrieval → passage set → generation → response. A minimal diagram:

```
User Query
   │
   ▼
[Retriever] → [Document Store] → [Top‑k Docs]
   │
   ▼
[Generator] (LLM + Docs) → Answer
```

## Why RAG Outperforms Vanilla LLMs

RAG augments a language model with a retrieval step, enabling it to consult external documents during generation. This design yields several practical advantages.

-

## Quick RAG Demo with OpenAI and Pinecone

This demo shows how to build a minimal Retrieval‑Augmented Generation pipeline with Pinecone and OpenAI’s GPT‑4o. First, create a Pinecone index and push a handful of short documents. Next, query the index with a user prompt, retrieve the top‑k snippets, and concatenate them into a context string that is sent to GPT‑4o. Finally, expose the whole flow behind a Flask route so you can hit it with a simple HTTP request.

```python
import os
from flask import Flask, request, jsonify
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

app = Flask(__name__)

# 1️⃣ Set up Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-demo"
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=1536,  # GPT‑4o embedding size
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# 2️⃣ Ingest sample docs
docs = [
    {"id": "1",
