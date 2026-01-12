"""
Database connection and schema setup for storing PDF chunks and embeddings
"""
import os
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import sql
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Create and return a database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "ai_learning"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise

def check_pgvector_available(cursor):
    """Check if pgvector extension is available"""
    try:
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        return cursor.fetchone() is not None
    except:
        return False

def create_tables():
    """Create the necessary tables for storing PDF chunks and embeddings"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if pgvector is available
        has_pgvector = check_pgvector_available(cursor)
        
        if has_pgvector:
            # Try to create extension if it doesn't exist
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
            except psycopg2.Error:
                pass  # Extension might already exist or not available
        
        # Check again after attempting to create
        has_pgvector = check_pgvector_available(cursor)
        
        if has_pgvector:
            # Use VECTOR type if pgvector is available
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pdf_chunks (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(filename, page_number, chunk_index)
                );
            """)
            
            # Create vector index
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embedding_vector 
                    ON pdf_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            except psycopg2.Error as e:
                print(f"Note: Could not create vector index: {e}")
        else:
            # Use REAL[] array type if pgvector is not available
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pdf_chunks (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding REAL[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(filename, page_number, chunk_index)
                );
            """)
            print("Note: pgvector extension not available. Using REAL[] array type.")
            print("      Vector similarity search will use manual cosine similarity.")
        
        # Create index on filename for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_filename ON pdf_chunks(filename);
        """)
        
        # Create pdf_documents table to store PDF metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL UNIQUE,
                display_name VARCHAR(255),
                description TEXT,
                options JSONB,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunk_count INTEGER DEFAULT 0
            );
        """)
        
        # Create index on filename for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pdf_documents_filename ON pdf_documents(filename);
        """)
        
        conn.commit()
        print("Database tables created successfully!")
        
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error creating tables: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def insert_or_update_pdf_document(filename: str, chunk_count: int, display_name: str = None, description: str = None, options: dict = None):
    """
    Insert or update PDF document metadata
    
    Args:
        filename: PDF filename
        chunk_count: Number of chunks for this PDF
        display_name: Optional display name
        description: Optional description
        options: Optional JSON options
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        import json
        options_json = json.dumps(options) if options else None
        
        # Use INSERT ... ON CONFLICT to update if exists
        cursor.execute("""
            INSERT INTO pdf_documents (filename, display_name, description, options, chunk_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (filename) 
            DO UPDATE SET 
                chunk_count = EXCLUDED.chunk_count,
                updated_at = CURRENT_TIMESTAMP,
                display_name = COALESCE(EXCLUDED.display_name, pdf_documents.display_name),
                description = COALESCE(EXCLUDED.description, pdf_documents.description),
                options = COALESCE(EXCLUDED.options, pdf_documents.options)
        """, (filename, display_name, description, options_json, chunk_count))
        
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error inserting/updating PDF document: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_all_pdf_documents():
    """
    Get all PDF documents with their metadata
    
    Returns:
        List of dictionaries with PDF document information
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT filename, display_name, description, options, uploaded_at, chunk_count
            FROM pdf_documents
            ORDER BY uploaded_at DESC
        """)
        
        results = cursor.fetchall()
        pdfs = []
        
        for row in results:
            import json
            pdfs.append({
                'filename': row[0],
                'display_name': row[1] or row[0],  # Use filename if display_name is None
                'description': row[2],
                'options': json.loads(row[3]) if row[3] else {},
                'uploaded_at': row[4].isoformat() if row[4] else None,
                'chunk_count': row[5] or 0
            })
        
        return pdfs
    except psycopg2.Error as e:
        print(f"Error fetching PDF documents: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def insert_chunks(chunks_data):
    """
    Get all PDF documents with their metadata
    
    Returns:
        List of dictionaries with PDF document information
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT filename, display_name, description, options, uploaded_at, chunk_count
            FROM pdf_documents
            ORDER BY uploaded_at DESC
        """)
        
        results = cursor.fetchall()
        pdfs = []
        
        for row in results:
            import json
            pdfs.append({
                'filename': row[0],
                'display_name': row[1] or row[0],  # Use filename if display_name is None
                'description': row[2],
                'options': json.loads(row[3]) if row[3] else {},
                'uploaded_at': row[4].isoformat() if row[4] else None,
                'chunk_count': row[5] or 0
            })
        
        return pdfs
    except psycopg2.Error as e:
        print(f"Error fetching PDF documents: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def insert_chunks(chunks_data):
    """
    Insert chunks with embeddings into the database
    
    Args:
        chunks_data: List of dictionaries with keys:
            - filename: str
            - page_number: int
            - chunk_index: int
            - chunk_text: str
            - embedding: list of floats (1536 dimensions)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if pgvector is available to format embedding correctly
        has_pgvector = check_pgvector_available(cursor)
        
        # Insert chunks one by one to handle different embedding formats
        inserted_count = 0
        for chunk in chunks_data:
            try:
                embedding = chunk['embedding']
                
                if has_pgvector:
                    # For pgvector, cast to vector type
                    cursor.execute("""
                        INSERT INTO pdf_chunks (filename, page_number, chunk_index, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        ON CONFLICT (filename, page_number, chunk_index) 
                        DO UPDATE SET 
                            chunk_text = EXCLUDED.chunk_text,
                            embedding = EXCLUDED.embedding
                    """, (
                        chunk['filename'],
                        chunk['page_number'],
                        chunk['chunk_index'],
                        chunk['chunk_text'],
                        str(embedding)  # Convert list to string for pgvector
                    ))
                else:
                    # For REAL[] array, pass as list directly
                    cursor.execute("""
                        INSERT INTO pdf_chunks (filename, page_number, chunk_index, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (filename, page_number, chunk_index) 
                        DO UPDATE SET 
                            chunk_text = EXCLUDED.chunk_text,
                            embedding = EXCLUDED.embedding
                    """, (
                        chunk['filename'],
                        chunk['page_number'],
                        chunk['chunk_index'],
                        chunk['chunk_text'],
                        embedding  # Pass as list for REAL[] array
                    ))
                
                inserted_count += 1
            except Exception as e:
                print(f"Error inserting chunk {chunk['filename']} page {chunk['page_number']} chunk {chunk['chunk_index']}: {e}")
                continue
        
        conn.commit()
        print(f"Successfully inserted {inserted_count}/{len(chunks_data)} chunks into database")
        
        # Update PDF document metadata
        if chunks_data:
            # Get unique filenames and their chunk counts
            from collections import Counter
            filename_counts = Counter(chunk['filename'] for chunk in chunks_data)
            
            for filename, count in filename_counts.items():
                try:
                    insert_or_update_pdf_document(filename, count)
                except Exception as e:
                    print(f"Warning: Could not update PDF document metadata for {filename}: {e}")
        
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error inserting chunks: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def search_similar_chunks(query_embedding, filename=None, limit=5):
    """
    Search for similar chunks using cosine similarity
    
    Args:
        query_embedding: List of floats representing the query embedding
        filename: Optional filename to filter by
        limit: Number of results to return
    
    Returns:
        List of tuples: (chunk_text, filename, page_number, similarity_score)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        has_pgvector = check_pgvector_available(cursor)
        
        if has_pgvector:
            # Use pgvector operators for fast similarity search
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            if filename:
                query = """
                    SELECT chunk_text, filename, page_number, 
                           1 - (embedding <=> %s::vector) as similarity
                    FROM pdf_chunks
                    WHERE filename = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cursor.execute(query, (embedding_str, filename, embedding_str, limit))
            else:
                query = """
                    SELECT chunk_text, filename, page_number,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM pdf_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                cursor.execute(query, (embedding_str, embedding_str, limit))
            
            results = cursor.fetchall()
            return results
        else:
            # Use manual cosine similarity
            return search_similar_chunks_manual(query_embedding, filename, limit, conn)
            
    except psycopg2.Error as e:
        # If vector operations fail, try manual cosine similarity
        print(f"Vector operation failed, trying manual cosine similarity: {e}")
        return search_similar_chunks_manual(query_embedding, filename, limit, conn)
    finally:
        cursor.close()
        conn.close()

def search_similar_chunks_manual(query_embedding, filename, limit, conn=None):
    """Fallback method using manual cosine similarity calculation"""
    import numpy as np
    
    should_close_conn = False
    if conn is None:
        conn = get_db_connection()
        should_close_conn = True
    cursor = conn.cursor()
    
    try:
        # Get all chunks
        if filename:
            cursor.execute("""
                SELECT id, chunk_text, filename, page_number, embedding
                FROM pdf_chunks
                WHERE filename = %s
            """, (filename,))
        else:
            cursor.execute("""
                SELECT id, chunk_text, filename, page_number, embedding
                FROM pdf_chunks
            """)
        
        chunks = cursor.fetchall()
        
        # Calculate cosine similarity manually
        query_vec = np.array(query_embedding)
        similarities = []
        
        for chunk_id, chunk_text, chunk_filename, page_number, embedding_data in chunks:
            # Handle different embedding formats
            if isinstance(embedding_data, list):
                embedding = np.array(embedding_data)
            elif isinstance(embedding_data, str):
                # Parse string representation
                embedding = np.array([float(x) for x in embedding_data.strip('{}[]').split(',')])
            else:
                # Assume it's already an array-like
                embedding = np.array(embedding_data)
            
            # Calculate cosine similarity
            dot_product = np.dot(query_vec, embedding)
            norm_query = np.linalg.norm(query_vec)
            norm_embedding = np.linalg.norm(embedding)
            
            if norm_query == 0 or norm_embedding == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_embedding)
            
            similarities.append((chunk_text, chunk_filename, page_number, float(similarity)))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[3], reverse=True)
        return similarities[:limit]
    finally:
        cursor.close()
        if should_close_conn and conn:
            conn.close()

def search_keywords_fulltext(query: str, filename: Optional[str] = None, limit: int = 10):
    """
    Search chunks using PostgreSQL full-text search (if available).
    
    Args:
        query: Query string
        filename: Optional filename filter
        limit: Number of results to return
    
    Returns:
        List of tuples: (chunk_text, filename, page_number, rank_score)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if full-text search is available (PostgreSQL always has it)
        # Create tsvector column and index if they don't exist
        try:
            # Add tsvector column if it doesn't exist
            cursor.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='pdf_chunks' AND column_name='chunk_text_tsvector'
                    ) THEN
                        ALTER TABLE pdf_chunks ADD COLUMN chunk_text_tsvector tsvector;
                    END IF;
                END $$;
            """)
            
            # Create index if it doesn't exist
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_text_tsvector 
                ON pdf_chunks USING gin(chunk_text_tsvector);
            """)
            
            # Update tsvector for all rows (only if column was just added)
            cursor.execute("""
                UPDATE pdf_chunks 
                SET chunk_text_tsvector = to_tsvector('english', chunk_text)
                WHERE chunk_text_tsvector IS NULL;
            """)
            
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            # If full-text search setup fails, return empty results
            # The hybrid_search module will fall back to simple keyword matching
            return []
        
        # Prepare query for full-text search
        # Convert query to tsquery format
        query_terms = query.split()
        tsquery = ' & '.join([term + ':*' for term in query_terms])  # Prefix matching
        
        # Build SQL query
        if filename:
            sql_query = """
                SELECT chunk_text, filename, page_number,
                       ts_rank(chunk_text_tsvector, to_tsquery('english', %s)) as rank
                FROM pdf_chunks
                WHERE chunk_text_tsvector @@ to_tsquery('english', %s)
                  AND filename = %s
                ORDER BY rank DESC
                LIMIT %s
            """
            cursor.execute(sql_query, (tsquery, tsquery, filename, limit))
        else:
            sql_query = """
                SELECT chunk_text, filename, page_number,
                       ts_rank(chunk_text_tsvector, to_tsquery('english', %s)) as rank
                FROM pdf_chunks
                WHERE chunk_text_tsvector @@ to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            """
            cursor.execute(sql_query, (tsquery, tsquery, limit))
        
        results = cursor.fetchall()
        
        # Normalize rank scores to 0-1 range
        # ts_rank returns values that can be > 1, so we normalize
        if results:
            max_rank = max(r[3] for r in results) if results else 1.0
            normalized_results = [
                (r[0], r[1], r[2], min(1.0, r[3] / max_rank) if max_rank > 0 else 0.0)
                for r in results
            ]
            return normalized_results
        
        return []
        
    except psycopg2.Error as e:
        # If full-text search fails, return empty (will fall back to simple matching)
        return []
    finally:
        cursor.close()
        conn.close()

