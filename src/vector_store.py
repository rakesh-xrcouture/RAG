import os
import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from src.config import config

# Updated Pinecone imports per the new SDK documentation
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class OptimizedHuggingFaceEmbeddings(Embeddings):
    """LangChain compatible embeddings class with batching and memory optimization"""
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=8):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = 768  # all-mpnet-base-v2 embedding dimension
        self.device = 'cpu'
        
        # Load tokenizer immediately, defer model loading until needed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
    
    def _load_model_if_needed(self):
        """Lazy-load the model to save memory when not in use"""
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts):
        """Create embeddings for documents using batching"""
        self._load_model_if_needed()
        all_embeddings = []
        
        # Process in batches to reduce memory usage
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
            
            # Free memory after each batch
            del inputs, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.vstack(all_embeddings).tolist()  # LangChain expects list of lists
    
    def embed_query(self, text):
        """Embed a single query using document batching method"""
        return self.embed_documents([text])[0]


class VectorStore:
    def __init__(self, batch_size=4):
        # Use optimized custom embeddings that implement LangChain's Embeddings interface
        self.embeddings = OptimizedHuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            batch_size=batch_size
        )
        # Initialize the updated Pinecone client via gRPC
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.INDEX_NAME
    
    def create_index(self):
        # Use the simpler has_index check instead of listing all indexes
        if self.index_name not in self.pc.list_indexes():
            print(f"Creating new Pinecone serverless index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Dimension matching your embedding model
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        else:
            print(f"Using existing Pinecone index: {self.index_name}")
    
    def upload_documents(self, documents, batch_size=50):
        if not documents:
            print("No documents to upload to Pinecone!")
            return None
        
        print(f"Uploading {len(documents)} documents to Pinecone in batches...")
        vectorstore = None
        
        # Process documents in smaller batches to manage memory usage
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            print(f"Processing batch {i//batch_size + 1}: documents {i+1} to {end_idx}")
            
            batch_docs = documents[i:end_idx]
            
            try:
                # For the first batch, create the vector store; then add documents subsequently
                if i == 0:
                    vectorstore = LangchainPinecone.from_documents(
                        batch_docs,
                        self.embeddings,
                        index_name=self.index_name
                    )
                else:
                    vectorstore.add_documents(batch_docs)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return vectorstore
    
    def load_index(self):
        # Check for the existence of the index using the new has_index method
        if self.index_name not in self.pc.list_indexes():
            print(f"Loading existing Pinecone index: {self.index_name}")
            return LangchainPinecone.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
        else:
            raise ValueError("Index does not exist. Please process documents first")
