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
    useFunctions: boolean = false
  ): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, {
      message,
      model,
      use_rag: useRag,
      filename: filename,
      use_functions: useFunctions
    } as ChatRequest);
  }
}

