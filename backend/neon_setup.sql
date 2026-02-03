-- Run this in Neon SQL Editor to enable pgvector and create tables for the AI Learning app.
-- Dashboard: https://console.neon.tech → your project → SQL Editor → New query → paste and Run.

-- 1. Enable pgvector extension (required for embeddings)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Table for PDF chunks and their embeddings (1536 = OpenAI text-embedding-3-small / ada-002 dimension)
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

CREATE INDEX IF NOT EXISTS idx_filename ON pdf_chunks(filename);

-- Vector index for similarity search (optional; if this fails on empty table, run it after uploading at least one PDF)
-- You can drop and recreate with larger lists later: DROP INDEX idx_embedding_vector; then CREATE INDEX ... WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_embedding_vector
ON pdf_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1);

-- 3. Table for PDF document metadata
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

CREATE INDEX IF NOT EXISTS idx_pdf_documents_filename ON pdf_documents(filename);
