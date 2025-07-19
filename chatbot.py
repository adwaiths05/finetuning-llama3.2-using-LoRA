from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

app = FastAPI(title="Ollama Chatbot", version="1.0")

# Allow CORS for HTML UI
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize Ollama model
llm = Ollama(model="llama3.2")

# Set up memory
memory = ConversationBufferMemory()

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "You’re a {tone} assistant. Conversation history: {history}\nUser: {input}\nAnswer in bullet points:"
)

# Create chain
chain = prompt | llm | StrOutputParser()

@app.post("/chat")
async def chat(data: dict):
    user_input = data.get("input", "")
    tone = data.get("tone", "friendly")
    history = memory.load_memory_variables({})['history']
    response = chain.invoke({"input": user_input, "tone": tone, "history": history})
    memory.save_context({"input": user_input}, {"output": response})
    return {"output": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

app = FastAPI(title="Ollama Chatbot", version="1.0")

# Allow CORS for HTML UI
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize Ollama model
llm = Ollama(model="llama3.2")

# Set up memory
memory = ConversationBufferMemory()

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "You’re a {tone} assistant. Conversation history: {history}\nUser: {input}\nAnswer in bullet points:"
)

# Create chain
chain = prompt | llm | StrOutputParser()

@app.post("/chat")
async def chat(data: dict):
    user_input = data.get("input", "")
    tone = data.get("tone", "friendly")
    history = memory.load_memory_variables({})['history']
    response = chain.invoke({"input": user_input, "tone": tone, "history": history})
    memory.save_context({"input": user_input}, {"output": response})
    return {"output": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
