import os
from src.config import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.chatbot import ChatBot

STATE_FILE = "processed_state.txt"

def initialize_system():
    print("Initializing the RAG chatbot system...")
    
    # Validate configuration
    config.validate()
    
    # Check if documents have already been processed
    if os.path.exists(STATE_FILE):
        print("Documents already processed. Loading existing vector store...")
        vector_store = VectorStore()
        vectorstore = vector_store.load_index()
        return ChatBot(vectorstore)
    
    # Process documents
    processor = DocumentProcessor()
    try:
        documents = processor.load_documents("data")
        processed_docs = processor.process_documents(documents)
    except FileNotFoundError:
        print("Please place your documents in the 'data' directory")
        return None
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return None
    
    # Set up vector store
    try:
        vector_store = VectorStore()
        vector_store.create_index()
        vectorstore = vector_store.upload_documents(processed_docs)
        
        # Save state to indicate documents have been processed
        with open(STATE_FILE, 'w') as f:
            f.write("processed")
    except Exception as e:
        print(f"Error setting up vector store: {str(e)}")
        return None
    
    return ChatBot(vectorstore)

def main():
    chatbot = initialize_system()
    if not chatbot:
        return
    
    print("\nChatbot is ready! Type 'exit' to end the conversation.")
    print("Type your questions about the documents...")
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        
        response = chatbot.chat(query)
        print(f"\nBot: {response}")

if __name__ == "__main__":
    main()