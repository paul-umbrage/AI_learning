"""
Helper script to set up the database
"""
from database import create_tables

if __name__ == "__main__":
    print("Setting up database...")
    try:
        create_tables()
        print("\n✅ Database setup complete!")
        print("\nYou can now ingest PDFs using:")
        print("  python ingest_pdf.py ../documents/your_file.pdf")
    except Exception as e:
        print(f"\n❌ Error setting up database: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. Database credentials in .env are correct")
        print("  3. Database 'ai_learning' exists (or create it manually)")

