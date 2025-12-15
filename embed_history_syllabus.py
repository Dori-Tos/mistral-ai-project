"""
Script to embed the complete Cambridge IGCSE History Option B syllabus
for historical fact-checking purposes.
"""

from aiFeatures.EmbeddingClient import get_embedding_client
import os

def embed_history_syllabus():
    """
    Embed the complete Cambridge IGCSE History syllabus with optimized parameters
    for fact-checking purposes.
    """
    print("="*60)
    print("Embedding Cambridge IGCSE History Option B: The 20th Century")
    print("="*60)
    
    # Get the embedding client
    client = get_embedding_client()
    print("✓ Client initialized successfully\n")
    
    # Path to the complete syllabus
    pdf_path = "History_syllabus/Cambridge_History_Option_B_the_20_th_century.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found.")
        return
    
    try:
        # Reset vector store for clean embedding
        print("Resetting vector store for fresh embedding...")
        client.reset_vector_store()
        print("✓ Vector store reset\n")
        
        # Load all documents from the PDF
        print(f"Loading documents from {pdf_path}...")
        documents = client.load_documents(pdf_path)
        print(f"✓ Loaded {len(documents)} pages\n")
        
        # Optimized parameters for historical fact-checking
        CHUNK_SIZE = 900          # Dense factual content - medium chunks
        CHUNK_OVERLAP = 180       # Good context continuity for events/dates
        
        print(f"Embedding parameters:")
        print(f"  - Chunk size: {CHUNK_SIZE} characters")
        print(f"  - Chunk overlap: {CHUNK_OVERLAP} characters")
        print(f"  - Model: {client.model_name}\n")
        
        # Process all documents
        all_chunks = []
        for idx, doc in enumerate(documents, 1):
            print(f"Processing page {idx}/{len(documents)}...", end="\r")
            chunks = client.split_document(
                doc, 
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            all_chunks.extend(chunks)
        
        print(f"\n✓ Split into {len(all_chunks)} total chunks\n")
        
        # Create embeddings with deduplication
        print("Creating embeddings (this may take a few minutes)...")
        vector_store = client.add_embeddings_with_deduplication(all_chunks)
        print("✓ Embeddings created successfully\n")
        
        # Show statistics
        stats = client.get_vector_store_stats()
        print("Vector Store Statistics:")
        print(f"  - Total documents: {stats['total_documents']}")
        print(f"  - Memory usage: {stats['memory_usage']}")
        print(f"  - Index size: {stats.get('index_size', 'N/A')}\n")
        
        # Save the vector store
        save_path = "RAG_vector_store"
        print(f"Saving vector store to {save_path}...")
        client.save_vector_store(save_path)
        print("✓ Vector store saved\n")
        
        # Test search functionality
        print("="*60)
        print("Testing Search Functionality")
        print("="*60)
        
        test_queries = [
            "Treaty of Versailles 1919",
            "causes of World War II",
            "Cold War origins",
            "League of Nations failure"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = client.search_similar(query, k=3)
            print(f"  Found {len(results)} relevant chunks")
            if results:
                # Show first result with metadata
                first_result = results[0]
                snippet = first_result.page_content[:150].replace('\n', ' ')
                metadata = first_result.metadata if hasattr(first_result, 'metadata') else {}
                page = metadata.get('page', 'N/A')
                filename = metadata.get('filename', 'N/A')
                print(f"  Top result from '{filename}', Page {page}:")
                print(f"    {snippet}...")
        
        print("\n" + "="*60)
        print("🎉 Syllabus embedding complete!")
        print("="*60)
        print("\nRecommended search parameters for fact-checking:")
        print("  - k=7-10 results for comprehensive context")
        print("  - threshold=0.65-0.75 for similarity matching")
        print("  - Use specific queries (dates, names, events)")
        
    except Exception as e:
        print(f"\n❌ Error during embedding: {e}")
        import traceback
        traceback.print_exc()


def test_fact_checking():
    """
    Test the fact-checking capability with sample queries.
    """
    print("\n" + "="*60)
    print("Fact-Checking Test")
    print("="*60)
    
    client = get_embedding_client()
    
    # Load the vector store
    try:
        client.load_vector_store("RAG_vector_store")
        print("✓ Vector store loaded\n")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print("Please run embed_history_syllabus() first.")
        return
    
    # Sample fact-checking queries
    fact_checks = [
        "When did World War I end?",
        "What were the main causes of the Great Depression?",
        "Who were the leaders during the Yalta Conference?",
        "What was the Marshall Plan?"
    ]
    
    for query in fact_checks:
        print(f"\nFact Check Query: '{query}'")
        print("-" * 60)
        
        # Use k=8 for comprehensive context
        results = client.search_similar(query, k=8)
        
        if results:
            print(f"Found {len(results)} relevant passages:\n")
            for i, doc in enumerate(results[:3], 1):
                content = doc.page_content.strip().replace('\n', ' ')[:200]
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                page = metadata.get('page', 'N/A')
                filename = metadata.get('filename', 'N/A')
                
                print(f"{i}. {content}...")
                print(f"   [Source: {filename}, Page {page}]")
        else:
            print("No relevant information found.")


if __name__ == "__main__":
    # Embed the complete syllabus
    embed_history_syllabus()
    
    # Test fact-checking
    print("\n")
    user_input = input("Would you like to test fact-checking? (y/n): ")
    if user_input.lower() == 'y':
        test_fact_checking()
