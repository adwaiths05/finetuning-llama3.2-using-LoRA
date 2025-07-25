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
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Ollama Chatbot with RAG and LoRA", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize base model and tokenizer
base_model = "llama3.2"
llm = Ollama(model=base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Load and prepare dataset
with open("dataset.json", "r") as f:
    dataset = json.load(f)

# Prepare dataset for fine-tuning
def prepare_dataset(dataset):
    formatted_data = []
    for entry in dataset:
        query = entry["query"]
        answer = entry["answer"]
        document = entry["documents"][0]["text"]
        formatted_data.append({
            "input": query,
            "output": f"Context: {document}\nAnswer: {answer}"
        })
    return formatted_data

train_data = prepare_dataset(dataset)

# Tokenize dataset
def tokenize_function(examples):
    inputs = [ex["input"] + "\n" + ex["output"] for ex in examples]
    tokenized = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = [tokenize_function([data]) for data in train_data]

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=2e-4,
    fp16=True,
    report_to="none"
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")

# Load fine-tuned model for inference
llm = Ollama(model="./lora_finetuned_model")

memory = ConversationBufferMemory()

# Initialize RAG components
embedder = SentenceTransformer("all-MiniLM-L6-v2")
documents = [entry["documents"][0]["text"] for entry in dataset]
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
