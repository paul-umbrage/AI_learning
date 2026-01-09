# AI Learning Project

This is a full-stack application for learning AI/ML concepts with OpenAI integration, built with Angular frontend and Python FastAPI backend.

## Project Structure

```
.
├── frontend/          # Angular application
├── backend/           # Python FastAPI application
└── README.md
```

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Python 3.8 or higher
- OpenAI API key

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the backend directory:
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

6. Run the backend server:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

   The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   
   Or:
   ```bash
   ng serve
   ```

   The frontend will be available at `http://localhost:4200`

## Usage

1. Make sure both backend and frontend servers are running
2. Open your browser and navigate to `http://localhost:4200`
3. Start chatting with the AI assistant!

## API Endpoints

### Backend API

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/chat` - Send a message to OpenAI
  - Request body:
    ```json
    {
      "message": "Your message here",
      "model": "gpt-3.5-turbo"  // optional, defaults to gpt-3.5-turbo
    }
    ```
  - Response:
    ```json
    {
      "response": "AI response here"
    }
    ```

## Development

### Backend
- The backend uses FastAPI with automatic API documentation
- Visit `http://localhost:8000/docs` for interactive API documentation
- Visit `http://localhost:8000/redoc` for alternative API documentation

### Frontend
- The frontend uses Angular with standalone components
- Hot reload is enabled by default
- The chat component is located in `src/app/components/chat/`

## Week 2: PDF Processing and Embeddings ✅

Week 2 is now implemented! See [WEEK2_README.md](WEEK2_README.md) for detailed instructions.

**Features:**
- PDF text extraction and chunking
- OpenAI embeddings generation
- PostgreSQL storage with vector similarity search
- Support for multiple PDFs with metadata (filename, page number)

**Quick Start:**
```bash
# Set up database
cd backend
python setup_database.py

# Ingest a PDF
python ingest_pdf.py ../documents/your_file.pdf
```

## Week 3-4: RAG & Function Calling ✅

Weeks 3 & 4 are now implemented! See [WEEK3_4_IMPLEMENTATION.md](WEEK3_4_IMPLEMENTATION.md) for feature details.

**Features:**
- RAG (Retrieval-Augmented Generation) with vector search
- Function Calling / Tool Calling with OpenAI
- Source citations and function call visualization
- Full-stack integration

**Quick Start:**
```bash
# See COMPLETE_RUN_GUIDE.md for detailed instructions
# Or follow the quick steps below
```

## Troubleshooting

### Backend Issues
- Make sure your OpenAI API key is correctly set in the `.env` file
- Check that port 8000 is not already in use
- Verify all Python dependencies are installed

### Frontend Issues
- Make sure the backend is running before starting the frontend
- Check browser console for CORS errors
- Verify the API URL in `chat.service.ts` matches your backend URL

## License

This project is for educational purposes.

