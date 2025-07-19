from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import json
import numpy as np

app = FastAPI(title="Ollama Chatbot with RAG", version="1.0")

# Allow CORS for HTML UI
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize Ollama model
llm = Ollama(model="llama3.2")

# Set up memory
memory = ConversationBufferMemory()

# Load dataset for RAG
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# Initialize embedding model for document retrieval
embedder = SentenceTransformer("all-MiniLM-L6-v2")
documents = [entry["documents"][0] for entry in dataset]
document_embeddings = embedder.encode(documents)

# Create prompt template with tone, history, and retrieved document
# Create prompt template with tone, history, and retrieved document
prompt = ChatPromptTemplate.from_template(
    "You’re a {tone} assistant. Conversation history: {history}\n"
    "Retrieved context: {context}\nUser: {input}\n"
    "Summarize your answer as a single, concise paragraph."
)


# Create chain
chain = prompt | llm | StrOutputParser()

# Retrieve top relevant document
def retrieve_document(query):
    query_embedding = embedder.encode(query)
    similarities = np.dot(document_embeddings, query_embedding) / (
        np.linalg.norm(document_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_idx = np.argmax(similarities)
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