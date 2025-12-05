"""
Test script to verify the MistralEmbedClient is working correctly.
Run this to test your embedding client with a sample PDF.
"""

from aiFeatures.EmbeddingClient import get_embedding_client, get_vector_store_manager

import os

def test_embedding_client():
    """Test the embedding client functionality."""
    print("Testing MistralEmbedClient...")
    
    # Get the client instance
    client = get_embedding_client()
    print("✓ Client initialized successfully")
    
    # Test with a sample PDF (you'll need to provide a PDF file)
    pdf_path = "History_syllabus/test.pdf"  # Change this to your PDF path
    
    if not os.path.exists(pdf_path):
        print(f"⚠️ Test file {pdf_path} not found.")
        print("Please place a PDF file in the test_files directory to test embedding functionality.")
        return
    
    try:
        # Load documents
        print(f"Loading documents from {pdf_path}...")
        documents = client.load_documents(pdf_path)
        print(f"✓ Loaded {len(documents)} documents")
        
        # Split first document
        if documents:
            print("Splitting first document...")
            texts = client.split_document(documents[0])
            print(f"✓ Split into {len(texts)} chunks")
            
            # Add embeddings with deduplication (safer)
            print("Creating embeddings with deduplication...")
            vector_store = client.add_embeddings_with_deduplication(texts)
            print(f"✓ Embeddings created successfully")
            
            # Show stats
            stats = client.get_vector_store_stats()
            print(f"✓ Vector store stats: {stats}")
            
            # Test search
            print("\nTesting search functionality...")
            results = client.search_similar("test query", k=3)
            print(f"✓ Search returned {len(results)} results")
            
            # Test saving (optional)
            save_path = "RAG_vector_store"
            print(f"Saving vector store to {save_path}...")
            client.save_vector_store(save_path)
            print("✓ Vector store saved")
            
            # Test loading (optional)
            print("Resetting and loading vector store...")
            client.reset_vector_store()
            client.load_vector_store(save_path)
            print("✓ Vector store loaded successfully")
            
            print("\n🎉 All tests passed! Your embedding client is ready to use.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Please check your .env file contains MISTRAL_API_KEY")
        
        
def test_vector_store_manager():
    """Test the vector store manager functionality."""
    print("Testing VectorStoreManager...")
    
    # Get the vector store manager instance
    vsm = get_vector_store_manager()
    print("✓ VectorStoreManager initialized successfully")
    
    # Further tests can be added here as needed

if __name__ == "__main__":
    test_embedding_client()
    test_vector_store_manager() 