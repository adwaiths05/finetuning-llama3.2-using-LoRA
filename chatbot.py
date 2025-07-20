from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss

app = FastAPI(title="Ollama Chatbot with RAG", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = Ollama(model="llama3.2")

memory = ConversationBufferMemory()

with open("dataset.json", "r") as f:
    dataset = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
documents = [entry["documents"][0] for entry in dataset]
document_embeddings = embedder.encode(documents)

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings.astype(np.float32))

prompt = ChatPromptTemplate.from_template(
    "You’re a {tone} assistant. Conversation history: {history}\n"
    "Retrieved context: {context}\nUser: {input}\n"
    "Summarize your answer as a single, concise paragraph."
)

chain = prompt | llm | StrOutputParser()

def retrieve_document(query):
    query_embedding = embedder.encode([query]).astype(np.float32)
    distances, indices = index.search(query_embedding, k=1)
    top_idx = indices[0][0]
    return documents[top_idx]

@app.post("/chat")
async def chat(data: dict):
    user_input = data.get("input", "")
    tone = data.get("tone", "friendly")
    history = memory.load_memory_variables({})['history']
    context = retrieve_document(user_input)
    response = chain.invoke({"input": user_input, "tone": tone, "history": history, "context": context})
    memory.save_context({"input": user_input}, {"output": response})
    return {"output": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
