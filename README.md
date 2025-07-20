# finetuning-llama3.2-using-RAG
## Overview
A **Retrieval-Augmented Generation (RAG)**-powered chatbot designed to answer questions about **AI in space technology** and **2025 missions**. This project combines scalable backend logic and a responsive frontend to deliver an engaging user experience.

---

## Features

- **RAG Implementation**: Utilizes FAISS for fast, relevant document retrieval from `dataset.json` (50 curated Q&A pairs).
- **Tone Selection**: Supports response styles – **Friendly**, **Formal**, or **Witty**.
- **Responsive Design**: Optimized for all devices using Bootstrap 5.
- **FastAPI Backend**: Chat API powered by LLaMA3.2 via Ollama with LangChain & FAISS.
- **Error Handling**: Detects and handles invalid inputs and backend issues gracefully.

---

## Tech Stack

- **Backend**: FastAPI, Ollama (llama3.2), LangChain, FAISS, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Frontend**: HTML, CSS (Orbitron font, neon theme), JavaScript, Bootstrap 5
- **Dataset**: `dataset.json` (50 curated Q&A pairs on AI in space tech)
