import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService, Source, FunctionCall } from '../../services/chat.service';

interface Message {
  text: string;
  isUser: boolean;
  timestamp: Date;
  sources?: Source[];
  functionCalls?: FunctionCall[];
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css'
})
export class ChatComponent {
  messages = signal<Message[]>([]);
  userMessage = '';
  isLoading = signal(false);
  error = signal<string | null>(null);
  
  // Feature toggles
  useRag = signal(true);
  useFunctions = signal(false);
  filenameFilter = '';
  
  // Advanced options
  showAdvancedOptions = signal(false);
  useReranking = true;
  rerankStrategy = 'combined';
  promptStrategy: string | undefined = undefined;
  useHybridSearch = false;
  vectorWeight = 0.7;
  keywordWeight = 0.3;
  useQueryExpansion = false;
  detectHallucinations = false;
  useLlmVerification = false;
  
  // Source expansion state
  expandedSources = new Set<number>();
  
  // Reranking strategies
  rerankStrategies = ['combined', 'threshold', 'keyword', 'diversity', 'length'];
  
  // Prompt strategies
  promptStrategies = [
    { value: undefined, label: 'Auto-select' },
    { value: 'strict', label: 'Strict' },
    { value: 'conversational', label: 'Conversational' },
    { value: 'technical', label: 'Technical' },
    { value: 'summarize', label: 'Summarize' },
    { value: 'qna', label: 'Q&A' }
  ];

  constructor(private chatService: ChatService) {
    // Add welcome message
    this.messages.set([{
      text: 'Hello! I\'m your AI assistant. How can I help you today?',
      isUser: false,
      timestamp: new Date()
    }]);
  }

  sendMessage() {
    const message = this.userMessage.trim();
    if (!message || this.isLoading()) {
      return;
    }

    // Add user message
    this.messages.update(msgs => [...msgs, {
      text: message,
      isUser: true,
      timestamp: new Date()
    }]);

    // Clear input
    this.userMessage = '';
    this.isLoading.set(true);
    this.error.set(null);

    // Send to backend
    this.chatService.sendMessage(
      message,
      'gpt-3.5-turbo',
      this.useRag(),
      this.filenameFilter.trim() || undefined,
      this.useFunctions(),
      {
        use_reranking: this.useReranking,
        rerank_strategy: this.rerankStrategy,
        prompt_strategy: this.promptStrategy,
        use_hybrid_search: this.useHybridSearch,
        vector_weight: this.vectorWeight,
        keyword_weight: this.keywordWeight,
        use_query_expansion: this.useQueryExpansion,
        detect_hallucinations: this.detectHallucinations,
        use_llm_verification: this.useLlmVerification
      }
    ).subscribe({
      next: (response) => {
        this.messages.update(msgs => [...msgs, {
          text: response.response,
          isUser: false,
          timestamp: new Date(),
          sources: response.sources,
          functionCalls: response.function_calls
        }]);
        this.isLoading.set(false);
      },
      error: (err) => {
        this.error.set(err.error?.detail || 'Failed to get response from AI');
        this.isLoading.set(false);
      }
    });
  }

  onKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  toggleRag() {
    this.useRag.update(value => !value);
  }

  toggleFunctions() {
    this.useFunctions.update(value => !value);
  }

  onFilenameChange(value: string) {
    this.filenameFilter = value;
  }

  clearFilenameFilter() {
    this.filenameFilter = '';
  }

  toggleSourcePreview(chunkIndex: number) {
    if (this.expandedSources.has(chunkIndex)) {
      this.expandedSources.delete(chunkIndex);
    } else {
      this.expandedSources.add(chunkIndex);
    }
  }

  isSourceExpanded(chunkIndex: number): boolean {
    return this.expandedSources.has(chunkIndex);
  }

  toggleAdvancedOptions() {
    this.showAdvancedOptions.update(value => !value);
  }

  updateVectorWeight(value: string) {
    const numValue = parseFloat(value);
    this.vectorWeight = numValue;
    this.keywordWeight = 1 - numValue;
  }

  updateKeywordWeight(value: string) {
    const numValue = parseFloat(value);
    this.keywordWeight = numValue;
    this.vectorWeight = 1 - numValue;
  }
}

