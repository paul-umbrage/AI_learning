import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Pool } from 'pg';

@Injectable()
export class DatabaseService {
  private pool: Pool;

  constructor(private configService: ConfigService) {
    this.pool = new Pool({
      host: this.configService.get<string>('DB_HOST', 'localhost'),
      port: this.configService.get<number>('DB_PORT', 5432),
      database: this.configService.get<string>('DB_NAME', 'ai_learning'),
      user: this.configService.get<string>('DB_USER', 'postgres'),
      password: this.configService.get<string>('DB_PASSWORD', 'postgres'),
    });
  }

  async checkPgVectorAvailable(): Promise<boolean> {
    try {
      const result = await this.pool.query(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector';"
      );
      return result.rows.length > 0;
    } catch {
      return false;
    }
  }

  async searchSimilarChunks(
    queryEmbedding: number[],
    filename?: string,
    limit: number = 5,
  ): Promise<Array<{ chunk_text: string; filename: string; page_number: number; similarity: number }>> {
    try {
      const hasPgVector = await this.checkPgVectorAvailable();
      const embeddingStr = '[' + queryEmbedding.join(',') + ']';

      if (hasPgVector) {
        // Use pgvector operators
        let query: string;
        let params: any[];

        if (filename) {
          query = `
            SELECT chunk_text, filename, page_number, 
                   1 - (embedding <=> $1::vector) as similarity
            FROM pdf_chunks
            WHERE filename = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
          `;
          params = [embeddingStr, filename, limit];
        } else {
          query = `
            SELECT chunk_text, filename, page_number,
                   1 - (embedding <=> $1::vector) as similarity
            FROM pdf_chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2
          `;
          params = [embeddingStr, limit];
        }

        const result = await this.pool.query(query, params);
        return result.rows.map((row) => ({
          chunk_text: row.chunk_text,
          filename: row.filename,
          page_number: parseInt(row.page_number),
          similarity: parseFloat(row.similarity),
        }));
      } else {
        // Manual cosine similarity
        return this.searchSimilarChunksManual(queryEmbedding, filename, limit);
      }
    } catch (error) {
      console.error('Vector operation failed, trying manual cosine similarity:', error);
      return this.searchSimilarChunksManual(queryEmbedding, filename, limit);
    }
  }

  private async searchSimilarChunksManual(
    queryEmbedding: number[],
    filename?: string,
    limit: number = 5,
  ): Promise<Array<{ chunk_text: string; filename: string; page_number: number; similarity: number }>> {
    let query: string;
    let params: any[];

    if (filename) {
      query = `
        SELECT id, chunk_text, filename, page_number, embedding
        FROM pdf_chunks
        WHERE filename = $1
      `;
      params = [filename];
    } else {
      query = `
        SELECT id, chunk_text, filename, page_number, embedding
        FROM pdf_chunks
      `;
      params = [];
    }

    const result = await this.pool.query(query, params);
    const chunks = result.rows;

    // Calculate cosine similarity
    const queryVec = queryEmbedding;
    const similarities: Array<{
      chunk_text: string;
      filename: string;
      page_number: number;
      similarity: number;
    }> = [];

    for (const chunk of chunks) {
      let embedding: number[];
      
      if (Array.isArray(chunk.embedding)) {
        embedding = chunk.embedding;
      } else if (typeof chunk.embedding === 'string') {
        embedding = chunk.embedding
          .replace(/[{}[\]]/g, '')
          .split(',')
          .map((x) => parseFloat(x.trim()));
      } else {
        embedding = Object.values(chunk.embedding) as number[];
      }

      // Calculate cosine similarity
      const dotProduct = queryVec.reduce((sum, val, i) => sum + val * embedding[i], 0);
      const normQuery = Math.sqrt(queryVec.reduce((sum, val) => sum + val * val, 0));
      const normEmbedding = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));

      const similarity =
        normQuery === 0 || normEmbedding === 0
          ? 0.0
          : dotProduct / (normQuery * normEmbedding);

      similarities.push({
        chunk_text: chunk.chunk_text,
        filename: chunk.filename,
        page_number: parseInt(chunk.page_number),
        similarity: similarity,
      });
    }

    // Sort by similarity and return top results
    similarities.sort((a, b) => b.similarity - a.similarity);
    return similarities.slice(0, limit);
  }

  async close(): Promise<void> {
    await this.pool.end();
  }
}

