"""
Migration script to populate pdf_documents table from existing pdf_chunks
Run this once to migrate existing data
"""
from database import get_db_connection, insert_or_update_pdf_document

def migrate_existing_pdfs():
    """Migrate existing PDFs from pdf_chunks to pdf_documents table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all unique filenames and their chunk counts
        cursor.execute("""
            SELECT filename, COUNT(*) as chunk_count
            FROM pdf_chunks
            GROUP BY filename
        """)
        
        results = cursor.fetchall()
        
        migrated_count = 0
        for filename, chunk_count in results:
            try:
                insert_or_update_pdf_document(filename, chunk_count)
                migrated_count += 1
                print(f"Migrated: {filename} ({chunk_count} chunks)")
            except Exception as e:
                print(f"Error migrating {filename}: {e}")
        
        print(f"\n✅ Migration complete! Migrated {migrated_count} PDF documents.")
        
    except Exception as e:
        print(f"Error during migration: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    print("Migrating existing PDFs to pdf_documents table...")
    migrate_existing_pdfs()
