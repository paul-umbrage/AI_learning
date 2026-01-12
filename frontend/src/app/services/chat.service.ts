import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Source {
  chunk_index: number;
  filename: string;
  page_number: number;
  similarity: number;
  text_preview: string;
}

export interface FunctionCall {
  function: string;
  arguments: any;
  result: string;
}

export interface ChatRequest {
  message: string;
  model?: string;
  use_rag?: boolean;
  filename?: string;
  use_functions?: boolean;
  use_reranking?: boolean;
  rerank_strategy?: string;
  prompt_strategy?: string;
  use_hybrid_search?: boolean;
  vector_weight?: number;
  keyword_weight?: number;
  use_query_expansion?: boolean;
  detect_hallucinations?: boolean;
  use_llm_verification?: boolean;
}

export interface ChatResponse {
  response: string;
  sources?: Source[];
  function_calls?: FunctionCall[];
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  sendMessage(
    message: string,
    model: string = 'gpt-3.5-turbo',
    useRag: boolean = true,
    filename?: string,
    useFunctions: boolean = false,
    options?: {
      use_reranking?: boolean;
      rerank_strategy?: string;
      prompt_strategy?: string;
      use_hybrid_search?: boolean;
      vector_weight?: number;
      keyword_weight?: number;
      use_query_expansion?: boolean;
      detect_hallucinations?: boolean;
      use_llm_verification?: boolean;
    }
  ): Observable<ChatResponse> {
    const request: ChatRequest = {
      message,
      model,
      use_rag: useRag,
      filename: filename,
      use_functions: useFunctions,
      ...options
    };
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, request);
  }

  getPdfs(): Observable<{ pdfs: string[], pdf_documents?: any[] }> {
    return this.http.get<{ pdfs: string[], pdf_documents?: any[] }>(`${this.apiUrl}/pdfs`);
  }
}

