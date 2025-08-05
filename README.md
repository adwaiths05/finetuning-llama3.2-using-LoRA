# Finetuning LLaMA3.2 with LoRA

## Overview
An AI powered chatbot fine-tuned with **LoRA** to answer questions about **AI in space technology**. Combines scalable backend logic with a responsive frontend for an engaging user experience.

---

## Features

- **LoRA Fine-Tuning**: Enhances LLaMA3.2 with low-rank adaptation for efficient training on space tech dataset.
- **Tone Selection**: Supports **Friendly**, **Formal**, or **Witty** response styles.
- **Responsive Design**: Optimized for all devices using Bootstrap 5.
- **Error Handling**: Gracefully manages invalid inputs and backend issues.

---

## Tech Stack

- **Backend**: FastAPI, Ollama (LLaMA3.2), LangChain, FAISS, Sentence-Transformers (`all-MiniLM-L6-v2`), PEFT (LoRA)
- **Frontend**: HTML, CSS (Orbitron font, neon theme), JavaScript, Bootstrap 5
- **Dataset**: `dataset.json` (50 curated Q&A pairs on AI in space tech)

---
