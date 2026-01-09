"""
Main script for ingesting PDF files: extracting text, chunking, generating embeddings, and storing in database
"""
import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from pdf_processor import process_pdf_to_chunks
from database import create_tables, insert_chunks, get_db_connection

load_dotenv()

def get_embeddings(chunks: list, api_key: str) -> list:
    """
    Generate embeddings for text chunks using OpenAI API
    
    Args:
        chunks: List of dictionaries with 'chunk_text' key
        api_key: OpenAI API key
    
    Returns:
        List of dictionaries with added 'embedding' key
    """
    client = OpenAI(api_key=api_key)
    
    chunks_with_embeddings = []
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        try:
            # Call OpenAI Embeddings API
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk['chunk_text']
            )
            
            embedding = response.data[0].embedding
            
            chunk_with_embedding = {
                'filename': chunk['filename'],
                'page_number': chunk['page_number'],
                'chunk_index': chunk['chunk_index'],
                'chunk_text': chunk['chunk_text'],
                'embedding': embedding
            }
            
            chunks_with_embeddings.append(chunk_with_embedding)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks...")
                
        except Exception as e:
            print(f"Error generating embedding for chunk {i}: {e}")
            continue
    
    print(f"Successfully generated embeddings for {len(chunks_with_embeddings)} chunks")
    return chunks_with_embeddings

def save_chunks_to_json(chunks: list, output_path: str):
    """Save chunks to JSON file (without embeddings to keep file size manageable)"""
    chunks_for_json = [
        {
            'filename': chunk['filename'],
            'page_number': chunk['page_number'],
            'chunk_index': chunk['chunk_index'],
            'chunk_text': chunk['chunk_text']
        }
        for chunk in chunks
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_for_json, f, indent=2, ensure_ascii=False)
    
    print(f"Saved chunks to {output_path}")

def ingest_pdf(pdf_path: str, chunk_method: str = "tokens", 
               chunk_size: int = 500, overlap: int = 50,
               save_json: bool = True,
               use_contextual_chunking: bool = False,
               contextual_method: str = "semantic",
               similarity_threshold: float = 0.7):
    """
    Complete PDF ingestion pipeline:
    1. Extract and chunk PDF text
    2. Generate embeddings
    3. Store in database
    4. Optionally save chunks to JSON
    
    Args:
        pdf_path: Path to PDF file
        chunk_method: "tokens", "characters", "semantic", or "sentence"
        chunk_size: Size of chunks
        overlap: Overlap size
        save_json: Whether to save chunks to JSON file
        use_contextual_chunking: Enable contextual chunking (content-type aware)
        contextual_method: "semantic" or "sentence" (only if use_contextual_chunking=True)
        similarity_threshold: Threshold for semantic breaks (0.0-1.0)
    """
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Chunk method: {chunk_method}, Size: {chunk_size}, Overlap: {overlap}")
    if use_contextual_chunking:
        print(f"Contextual chunking: ENABLED (method: {contextual_method})")
    
    # Get OpenAI client for semantic chunking if needed
    client = None
    if use_contextual_chunking and contextual_method == "semantic":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            print("Warning: OPENAI_API_KEY not found. Falling back to sentence-based chunking.")
            contextual_method = "sentence"
    
    # Step 1: Extract and chunk PDF
    print("\nStep 1: Extracting and chunking PDF text...")
    chunks = process_pdf_to_chunks(
        pdf_path, 
        chunk_method, 
        chunk_size, 
        overlap,
        use_contextual_chunking=use_contextual_chunking,
        contextual_method=contextual_method,
        similarity_threshold=similarity_threshold,
        client=client
    )
    print(f"Created {len(chunks)} chunks from PDF")
    
    # Show content type distribution if contextual chunking was used
    if use_contextual_chunking:
        from collections import Counter
        content_types = Counter(chunk.get('content_type', 'unknown') for chunk in chunks)
        print(f"Content type distribution: {dict(content_types)}")
    
    # Step 2: Generate embeddings
    print("\nStep 2: Generating embeddings...")
    chunks_with_embeddings = get_embeddings(chunks, api_key)
    
    if not chunks_with_embeddings:
        print("Error: No embeddings generated")
        return
    
    # Step 3: Store in database
    print("\nStep 3: Storing in database...")
    try:
        insert_chunks(chunks_with_embeddings)
        print("Successfully stored chunks in database")
    except Exception as e:
        print(f"Error storing in database: {e}")
        print("Chunks and embeddings are still available in memory")
    
    # Step 4: Save to JSON (optional, without embeddings)
    if save_json:
        json_path = pdf_path.replace('.pdf', '_chunks.json')
        save_chunks_to_json(chunks, json_path)
    
    print("\n✅ PDF ingestion complete!")
    print(f"   - Total chunks: {len(chunks_with_embeddings)}")
    print(f"   - Stored in database: pdf_chunks table")
    if save_json:
        print(f"   - Chunks saved to: {json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py <path_to_pdf> [chunk_method] [chunk_size] [overlap] [use_contextual] [contextual_method] [similarity_threshold]")
        print("Example: python ingest_pdf.py documents/my_document.pdf tokens 500 50")
        print("Example: python ingest_pdf.py documents/my_document.pdf semantic 500 50 true semantic 0.7")
        print("\nParameters:")
        print("  chunk_method: 'tokens' (default), 'characters', 'semantic', or 'sentence'")
        print("  chunk_size: Size of chunks (default: 500 tokens or 1000 characters)")
        print("  overlap: Overlap between chunks (default: 50 tokens or 200 characters)")
        print("  use_contextual: 'true' to enable contextual chunking (default: false)")
        print("  contextual_method: 'semantic' or 'sentence' (default: semantic)")
        print("  similarity_threshold: Threshold for semantic breaks 0.0-1.0 (default: 0.7)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    chunk_method = sys.argv[2] if len(sys.argv) > 2 else "tokens"
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else (500 if chunk_method == "tokens" else 1000)
    overlap = int(sys.argv[4]) if len(sys.argv) > 4 else (50 if chunk_method == "tokens" else 200)
    use_contextual = sys.argv[5].lower() == "true" if len(sys.argv) > 5 else False
    contextual_method = sys.argv[6] if len(sys.argv) > 6 else "semantic"
    similarity_threshold = float(sys.argv[7]) if len(sys.argv) > 7 else 0.7
    
    # Ensure database tables exist
    print("Setting up database...")
    try:
        create_tables()
    except Exception as e:
        print(f"Warning: Could not create database tables: {e}")
        print("Make sure PostgreSQL is running and credentials are correct in .env file")
    
    # Run ingestion
    ingest_pdf(
        pdf_path, 
        chunk_method, 
        chunk_size, 
        overlap,
        use_contextual_chunking=use_contextual,
        contextual_method=contextual_method,
        similarity_threshold=similarity_threshold
    )

