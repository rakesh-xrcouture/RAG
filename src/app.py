# main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

from src.vector_store import VectorStore
from src.chatbot import ChatBot

# Define request and response models
class Query(BaseModel):
    text: str

class ChatResponse(BaseModel):
    response: str
    source_documents: Optional[list] = None

# Global chatbot instance
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the chatbot on startup - just load existing vector store
    global chatbot
    try:
        vector_store = VectorStore()
        vectorstore = vector_store.load_index()
        chatbot = ChatBot(vectorstore)
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
    yield
    # Clean up resources if needed on shutdown

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

def get_chatbot():
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot service unavailable")
    return chatbot

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running. Send POST requests to /chat endpoint."}

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query, bot: ChatBot = Depends(get_chatbot)):
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        response = bot.chat(query.text)
        
        # Assuming your ChatBot.chat() method returns just a string
        # If it returns more structured data, modify accordingly
        return ChatResponse(
            response=response,
            source_documents=None  # Add source documents if your chatbot provides them
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# For local development and testing
if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))