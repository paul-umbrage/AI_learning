# Documents Directory

Place your PDF files here for processing.

## Example Document

An example document is available for testing:

- **Text version**: `example_ai_learning.txt` - Ready to use, contains AI/ML educational content
- **PDF version**: Can be generated using the script in `backend/create_example_pdf.py`

The example document covers:
- Introduction to Artificial Intelligence
- Machine Learning concepts
- Natural Language Processing
- Retrieval-Augmented Generation (RAG)

## Usage

### Using the Example Document

1. **Convert text to PDF** (if needed):
   - Use any word processor (Word, Pages, Google Docs) to open `example_ai_learning.txt` and save as PDF
   - Or use online converters
   - Or use the Python script: `cd backend && python create_example_pdf.py ../documents/example_ai_learning.pdf`

2. **Ingest the PDF**:
   ```bash
   cd backend
   python ingest_pdf.py ../documents/example_ai_learning.pdf
   ```

### Using Your Own PDF

After placing a PDF file in this directory, run the ingestion script:

```bash
cd backend
python ingest_pdf.py ../documents/your_file.pdf
```

## Example Commands

```bash
# Basic ingestion
python ingest_pdf.py ../documents/research_paper.pdf

# With custom chunking parameters
python ingest_pdf.py ../documents/research_paper.pdf tokens 500 50
```

This will:
1. Extract text from the PDF
2. Chunk the text into manageable pieces
3. Generate embeddings using OpenAI API
4. Store everything in PostgreSQL database

