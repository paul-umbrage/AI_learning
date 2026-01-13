# AI Learning Project

A full-stack RAG (Retrieval-Augmented Generation) application for learning AI/ML concepts with OpenAI integration. Built with Angular frontend and Python FastAPI backend, featuring PDF document processing, vector embeddings, and intelligent chat with source citations.

## 🎯 Features

- **PDF Document Processing**: Upload and process PDF documents with intelligent chunking
- **Vector Embeddings**: Convert text into embeddings using OpenAI's embedding models
- **RAG (Retrieval-Augmented Generation)**: Enhance AI responses with relevant document context
- **Function Calling**: Extend AI capabilities with custom tools (weather, calculator, time)
- **Source Citations**: See exactly which documents and pages support each answer
- **Hybrid Search**: Combine vector similarity with keyword matching
- **Query Expansion**: Generate multiple query variations for better retrieval
- **Hallucination Detection**: Verify AI responses against source documents

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v18 or higher) - [Download](https://nodejs.org/)
- **Python** (3.8 or higher) - [Download](https://www.python.org/downloads/)
- **PostgreSQL** (12 or higher) OR **Docker** - [PostgreSQL Download](https://www.postgresql.org/download/) | [Docker Download](https://www.docker.com/get-started)
- **OpenAI API Key** - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **npm** (comes with Node.js)
- **pip** (comes with Python)

## 🚀 Quick Start (5 Minutes)

For the fastest setup, see [QUICK_START.md](QUICK_START.md) for a condensed guide.

## 📦 Complete Setup Guide

### Step 1: Database Setup

You have two options for setting up the database:

#### Option A: Using Docker (Recommended - Easiest)

```bash
# Start PostgreSQL with pgvector extension
docker run --name ai-learning-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=ai_learning \
  -p 5432:5432 \
  -d pgvector/pgvector:pg15

# Verify it's running
docker ps
```

**Note:** If port 5432 is already in use, you can change it (e.g., `-p 5433:5432`) and update the `DB_PORT` in your `.env` file accordingly.

#### Option B: Local PostgreSQL Installation

1. Install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/)
2. Create the database:
   ```bash
   psql -U postgres
   CREATE DATABASE ai_learning;
   \q
   ```
3. Install pgvector extension (optional but recommended for better performance):
   ```bash
   # On macOS
   brew install pgvector
   
   # On Ubuntu/Debian
   sudo apt-get install postgresql-15-pgvector
   
   # Then enable it in the database
   psql -U postgres -d ai_learning
   CREATE EXTENSION IF NOT EXISTS vector;
   \q
   ```

### Step 2: Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment file:**
   ```bash
   cp env.example .env
   ```

5. **Edit `.env` file with your configuration:**
   ```bash
   # Required: OpenAI API Key
   OPENAI_API_KEY=sk-your-actual-api-key-here
   
   # Database Configuration
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=ai_learning
   DB_USER=postgres
   DB_PASSWORD=postgres
   ```

   **Important:** Update the database credentials if you're using different values (especially if using Docker with custom settings).

6. **Set up the database schema:**
   ```bash
   python setup_database.py
   ```

   This will create the necessary tables and indexes for storing PDF chunks and embeddings.

7. **Start the backend server:**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

   You should see:
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

   The backend will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - Alternative Docs: `http://localhost:8000/redoc`

### Step 3: Frontend Setup

1. **Open a new terminal and navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```
   
   Or:
   ```bash
   ng serve
   ```

   The frontend will be available at `http://localhost:4200`

### Step 4: Ingest Your First PDF (Optional)

To use RAG features, you need to upload some documents:

1. **Using the web interface:**
   - Navigate to `http://localhost:4200`
   - Use the upload feature in the chat interface

2. **Using the command line:**
   ```bash
   # From the backend directory (with venv activated)
   cd backend
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Ingest a PDF file
   python ingest_pdf.py ../documents/example_ai_learning.pdf
   
   # Or ingest from HTML
   python ingest_pdf.py ../documents/example_ai_learning.html
   ```

   **Note:** The first PDF ingestion may take a few minutes as it generates embeddings for each chunk.

## 🎮 Usage

1. **Start both servers:**
   - Backend: `http://localhost:8000` (Terminal 1)
   - Frontend: `http://localhost:4200` (Terminal 2)

2. **Open your browser:**
   - Navigate to `http://localhost:4200`

3. **Start chatting:**
   - **Without RAG**: Ask general questions (works without PDFs)
   - **With RAG**: Enable the "Use RAG" toggle and ask questions about your uploaded documents
   - **With Functions**: Enable "Use Functions" to allow the AI to call tools (weather, calculator, time)

4. **View sources:**
   - When RAG is enabled, you'll see source citations showing which documents and pages support each answer

## 📡 API Endpoints

### Backend API

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/pdfs` - Get list of uploaded PDFs
- `POST /api/chat` - Send a message to OpenAI
  - Request body:
    ```json
    {
      "message": "Your message here",
      "model": "gpt-3.5-turbo",
      "use_rag": true,
      "use_functions": false,
      "use_reranking": true,
      "use_hybrid_search": false,
      "filename": "optional-filter-by-filename.pdf"
    }
    ```
  - Response:
    ```json
    {
      "response": "AI response here",
      "sources": [
        {
          "chunk_index": 1,
          "filename": "document.pdf",
          "page_number": 5,
          "similarity": 0.89,
          "text_preview": "..."
        }
      ],
      "function_calls": []
    }
    ```
- `POST /api/upload` - Upload and ingest a PDF file

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger documentation where you can test all endpoints directly.

## 🏗️ Project Structure

```
.
├── backend/                 # Python FastAPI backend
│   ├── main.py             # Main API server
│   ├── database.py         # Database operations
│   ├── ingest_pdf.py       # PDF ingestion script
│   ├── pdf_processor.py    # PDF text extraction and chunking
│   ├── reranking.py        # Reranking strategies
│   ├── hybrid_search.py    # Hybrid search (vector + keyword)
│   ├── query_expansion.py  # Query expansion
│   ├── contextual_chunking.py  # Advanced chunking
│   ├── prompts/            # Prompt templates and strategies
│   ├── utils/              # Utilities (logging, error handling, etc.)
│   ├── evaluation/         # Evaluation metrics
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # Angular frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/
│   │   │   │   └── chat/   # Chat component
│   │   │   └── services/
│   │   │       └── chat.service.ts  # API service
│   │   └── main.ts
│   └── package.json
│
├── documents/              # Sample documents and PDFs
│   ├── example_ai_learning.pdf
│   └── animal_kingdom.pdf
│
└── README.md               # This file
```

## 🔧 Development

### Backend Development

- **Hot Reload**: The backend uses `--reload` flag for automatic restart on code changes
- **API Documentation**: Automatically generated at `/docs` and `/redoc`
- **Logging**: Structured logging with request tracking and cost calculation

### Frontend Development

- **Hot Reload**: Enabled by default with Angular dev server
- **Component Location**: `src/app/components/chat/`
- **Service Location**: `src/app/services/chat.service.ts`

## 🐛 Troubleshooting

### Backend Issues

**Backend won't start:**
- Check that port 8000 is not already in use: `lsof -i :8000` (macOS/Linux) or `netstat -ano | findstr :8000` (Windows)
- Verify your `.env` file exists and has a valid OpenAI API key
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check database connection: Verify PostgreSQL is running and credentials are correct

**Database connection errors:**
- Verify PostgreSQL is running: `docker ps` (if using Docker) or `pg_isready` (if local)
- Check database credentials in `.env` file match your PostgreSQL setup
- Ensure the database exists: `psql -U postgres -l` should show `ai_learning`
- For Docker: Make sure the container is running: `docker start ai-learning-db`

**OpenAI API errors:**
- Verify your API key is correct and has credits
- Check your OpenAI account usage limits
- Ensure the API key is set in `.env` file (not just `env.example`)

### Frontend Issues

**Frontend won't start:**
- Check that port 4200 is not already in use
- Make sure Node.js is installed: `node --version` (should be v18+)
- Try deleting `node_modules` and reinstalling: `rm -rf node_modules && npm install`

**Can't connect to backend:**
- Verify backend is running on port 8000: Visit `http://localhost:8000/health`
- Check browser console for CORS errors (should be handled automatically)
- Verify the API URL in `frontend/src/app/services/chat.service.ts` matches your backend URL

**CORS errors:**
- The backend is configured to allow `http://localhost:4200` by default
- If using a different port, update CORS settings in `backend/main.py`

### PDF Ingestion Issues

**PDF upload fails:**
- Ensure the file is a valid PDF
- Check file size (very large PDFs may take a long time)
- Verify database is accessible and tables exist: `python setup_database.py`
- Check backend logs for specific error messages

**No results from RAG:**
- Ensure at least one PDF has been ingested
- Check database has chunks: `psql -U postgres -d ai_learning -c "SELECT COUNT(*) FROM pdf_chunks;"`
- Verify embeddings were generated (check backend logs during ingestion)

## 📚 Additional Documentation

- **[QUICK_START.md](QUICK_START.md)** - 5-minute quick setup guide
- **[COMPLETE_ENVIRONMENT_SETUP.md](COMPLETE_ENVIRONMENT_SETUP.md)** - Complete environment setup instructions
- **[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)** - Comprehensive project explanation
- **[API_TESTING.md](API_TESTING.md)** - Testing APIs with payloads and responses

## 🧪 Testing

### Test the Setup

1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **List PDFs:**
   ```bash
   curl http://localhost:8000/api/pdfs
   ```

3. **Test Chat (without RAG):**
   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, how are you?"}'
   ```

### Run Evaluation Tests

```bash
cd backend
source venv/bin/activate
python test_quick_wins.py
```

## 🛠️ Utility Scripts

- **Clear all PDFs from database:**
  ```bash
  cd backend
  python clear_all_pdfs.py
  ```

- **Create example PDF:**
  ```bash
  cd backend
  python create_example_pdf.py ../documents/example.pdf
  ```

## 📝 Environment Variables

All environment variables are configured in `backend/.env`:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `DB_HOST` | PostgreSQL host | Yes |
| `DB_PORT` | PostgreSQL port | Yes (default: 5432) |
| `DB_NAME` | Database name | Yes (default: ai_learning) |
| `DB_USER` | Database user | Yes (default: postgres) |
| `DB_PASSWORD` | Database password | Yes |

## 🤝 Contributing

This is an educational project. Feel free to fork, modify, and experiment!

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- OpenAI for the API and embedding models
- FastAPI for the excellent Python web framework
- Angular for the frontend framework
- pgvector for PostgreSQL vector extension

---

**Need help?** Check the troubleshooting section above or review the additional documentation files.
