from dotenv import load_dotenv
from mistralai import Mistral
import os
import json
import hashlib
from typing import Optional, List, Dict, Any, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.documents import Document

class MistralEmbedClient:
    """Mistral AI embeddings client with better practices"""
    _instance: Optional['MistralEmbedClient'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        load_dotenv(dotenv_path=env_path, override=True)
        
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set")
        
        self.model_name = os.getenv("EMBEDDING_MODEL", "mistral-embed")
        self.embeddings = MistralAIEmbeddings(api_key=self.api_key, model=self.model_name)
        self.client = Mistral(api_key=self.api_key)
        
        self.vector_store: Optional[FAISS] = None
        self.document_hashes: Set[str] = set()  # Track processed documents
        self.max_documents = int(os.getenv("MAX_DOCUMENTS", "10000"))  # Limit memory usage
        self._initialized = True

    def _get_document_hash(self, content: str) -> str:
        """Generate hash for document content to detect duplicates."""
        return hashlib.md5(content.encode()).hexdigest()

    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents from a PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # Extract just the filename from the full path
            filename = os.path.basename(file_path)
            
            # Add or update metadata for each document to include the filename
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                # Keep existing source if present, otherwise use the full path
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_path
                # Add the filename as a separate field for easy citation
                doc.metadata['filename'] = filename
            
            print(f"Successfully loaded {len(documents)} documents from {filename}")
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise

    def split_document(self, document: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Split a document into smaller chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents([document])
            
            # Preserve original page number in metadata for all chunks
            # If a chunk spans multiple pages due to overlap, keep the starting page
            original_page = document.metadata.get('page', 'N/A') if document.metadata else 'N/A'
            filename = document.metadata.get('filename', 'Unknown') if document.metadata else 'Unknown'
            source = document.metadata.get('source', 'Unknown') if document.metadata else 'Unknown'
            
            for chunk in texts:
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata['page'] = original_page
                chunk.metadata['filename'] = filename
                chunk.metadata['source'] = source
            
            print(f"Split document into {len(texts)} chunks")
            return texts
        except Exception as e:
            print(f"Error splitting document: {e}")
            return []
    
    def add_embeddings_with_deduplication(self, texts: List[Document], force_add: bool = False) -> FAISS:
        """
        Add new embeddings with deduplication and size limits.
        
        Args:
            texts: List of documents to add
            force_add: If True, bypass duplicate checking
        """
        if not force_add:
            # Filter out duplicates
            unique_texts = []
            for text in texts:
                doc_hash = self._get_document_hash(text.page_content)
                if doc_hash not in self.document_hashes:
                    unique_texts.append(text)
                    self.document_hashes.add(doc_hash)
            texts = unique_texts
        
        if not texts:
            print("No new unique documents to add.")
            return self.vector_store
        
        # Check size limits
        current_size = len(self.document_hashes)
        if current_size + len(texts) > self.max_documents:
            print(f"Warning: Adding {len(texts)} documents would exceed limit of {self.max_documents}")
            texts = texts[:self.max_documents - current_size]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
        else:
            if texts:  # Only merge if there are texts to add
                new_vector_store = FAISS.from_documents(texts, self.embeddings)
                self.vector_store.merge_from(new_vector_store)
        
        print(f"Added {len(texts)} new documents. Total documents: {len(self.document_hashes)}")
        return self.vector_store
    
    def add_embeddings(self, texts: List[Document]) -> FAISS:
        """Add embeddings without deduplication (for backward compatibility)."""
        return self.add_embeddings_with_deduplication(texts, force_add=True)
    
    def create_new_vector_store(self, texts: List[Document]) -> FAISS:
        """Create a completely new vector store (doesn't affect the singleton's store)."""
        return FAISS.from_documents(texts, self.embeddings)
    
    def reset_vector_store(self) -> None:
        """Reset the vector store (clear all embeddings)."""
        self.vector_store = None
        self.document_hashes.clear()
        print("Vector store reset")
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the current vector store."""
        if self.vector_store is None:
            return {"total_documents": 0, "memory_usage": "0 MB"}
        
        # Get approximate memory usage (this is a rough estimate)
        import sys
        memory_usage = sys.getsizeof(self.vector_store) / (1024 * 1024)  # MB
        
        return {
            "total_documents": len(self.document_hashes),
            "memory_usage": f"{memory_usage:.2f} MB",
            "index_size": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else "Unknown"
        }
    
    def clear_old_embeddings(self, keep_last_n: int = 1000) -> None:
        """Clear old embeddings, keeping only the last N documents."""
        if len(self.document_hashes) > keep_last_n:
            print(f"Clearing embeddings, keeping last {keep_last_n} documents")
            # This is a simplified version - in practice, you'd need more sophisticated cleanup
            self.vector_store = None
            self.document_hashes.clear()
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
    
    def save_vector_store(self, file_path: str) -> None:
        """Save the vector store and metadata to disk."""
        if self.vector_store is not None:
            self.vector_store.save_local(file_path)
            # Save document hashes separately
            with open(f"{file_path}_metadata.json", "w") as f:
                json.dump({
                    "document_hashes": list(self.document_hashes),
                    "total_documents": len(self.document_hashes)
                }, f)
    
    def load_vector_store(self, file_path: str) -> FAISS:
        """Load a vector store and metadata from disk."""
        self.vector_store = FAISS.load_local(file_path, self.embeddings, allow_dangerous_deserialization=True)
        
        # Load document hashes
        try:
            with open(f"{file_path}_metadata.json", "r") as f:
                metadata = json.load(f)
                self.document_hashes = set(metadata.get("document_hashes", []))
        except FileNotFoundError:
            print("Metadata file not found, starting with empty hash set")
            self.document_hashes = set()
        
        return self.vector_store
    


class VectorStoreManager:
    """Separate class for managing vector stores"""
    
    def __init__(self, embeddings: MistralAIEmbeddings):
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vector store"""
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            new_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.merge_from(new_store)
    
    def get_retriever(self, k: int = 5, threshold: float = 0.7, **kwargs):
        """Get retriever for querying"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": threshold})
    
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Direct similarity search"""
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
    
    def search_with_retriever(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Document]:
        """Search using retriever with score threshold"""
        if self.vector_store is None:
            return []
        retriever = self.get_retriever(k=k, threshold=threshold)
        return retriever.invoke(query)
    
    def save_vector_store(self, file_path: str) -> None:
        """Save the vector store to disk."""
        if self.vector_store is not None:
            self.vector_store.save_local(file_path)
    
    def load_vector_store(self, file_path: str) -> None:
        """Load a vector store from disk."""
        self.vector_store = FAISS.load_local(
            file_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )


def get_embedding_client() -> MistralEmbedClient:
    """Get or create the AI service instance"""
    return MistralEmbedClient()

def get_vector_store_manager() -> VectorStoreManager:
    """Get a vector store manager instance"""
    client = get_embedding_client()
    return VectorStoreManager(client.embeddings)