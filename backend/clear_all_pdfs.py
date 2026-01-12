"""
Script to clear all PDFs from the database
This will remove all chunks and PDF document records
"""
from database import get_db_connection, create_tables

def clear_all_pdfs():
    """Clear all PDFs from both pdf_chunks and pdf_documents tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        print("Clearing all PDFs from database...")
        
        # Clear pdf_chunks table
        cursor.execute("DELETE FROM pdf_chunks")
        chunks_deleted = cursor.rowcount
        print(f"Deleted {chunks_deleted} chunks from pdf_chunks table")
        
        # Clear pdf_documents table (if it exists)
        try:
            cursor.execute("DELETE FROM pdf_documents")
            docs_deleted = cursor.rowcount
            print(f"Deleted {docs_deleted} records from pdf_documents table")
        except Exception as e:
            # Table doesn't exist, create it
            print("pdf_documents table doesn't exist. Creating it...")
            cursor.close()
            conn.close()
            create_tables()
            print("✅ Database tables created/updated!")
            return
        
        conn.commit()
        print("\n✅ All PDFs cleared from database!")
        print("\nYou can now re-upload PDFs using the upload page or:")
        print("  python ingest_pdf.py ../documents/your_file.pdf")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error clearing PDFs: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    response = input("⚠️  This will delete ALL PDFs from the database. Continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        clear_all_pdfs()
    else:
        print("Operation cancelled.")
