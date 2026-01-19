import { Component, signal, OnInit, OnDestroy } from '@angular/core';
import { Router, NavigationEnd } from '@angular/router';
import { filter } from 'rxjs/operators';
import { Subscription } from 'rxjs';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
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
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.css'
})
export class ChatComponent implements OnInit, OnDestroy {
  private routerSubscription?: Subscription;
  messages = signal<Message[]>([]);
  userMessage = '';
  isLoading = signal(false);
  error = signal<string | null>(null);

  // Feature toggles
  useRag = signal(true);
  useFunctions = signal(false);
  filenameFilter = '';
  availablePdfs: string[] = [];
  selectedPdf: string = '';
  isLoadingPdfs = false;
  selectedModel = 'gpt-3.5-turbo';

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

  // Available models
  availableModels = [
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
    { value: 'gpt-4', label: 'GPT-4' },
    { value: 'gpt-4-turbo-preview', label: 'GPT-4 Turbo' },
    { value: 'gpt-4o', label: 'GPT-4o' },
    { value: 'gpt-4o-mini', label: 'GPT-4o Mini' }
  ];

  // Copy feedback state
  copiedMessageIndex: number | null = null;

  constructor(private chatService: ChatService, private router: Router) {
    // Load chat history from localStorage
    this.loadChatHistory();
    
    // If no history, add welcome message
    if (this.messages().length === 0) {
      this.messages.set([{
        text: 'Hello! I\'m your AI assistant. How can I help you today?',
        isUser: false,
        timestamp: new Date()
      }]);
    }
  }

  ngOnInit() {
    this.loadPdfs();

    // Refresh PDF list when returning from upload page
    this.routerSubscription = this.router.events
      .pipe(filter(event => event instanceof NavigationEnd))
      .subscribe((event: any) => {
        if (event.url === '/') {
          this.loadPdfs();
        }
      });
  }

  ngOnDestroy() {
    if (this.routerSubscription) {
      this.routerSubscription.unsubscribe();
    }
  }

  loadPdfs() {
    this.isLoadingPdfs = true;
    this.chatService.getPdfs().subscribe({
      next: (response) => {
        // Use pdf_documents if available, otherwise fall back to simple pdfs list
        if (response.pdf_documents && response.pdf_documents.length > 0) {
          this.availablePdfs = response.pdf_documents.map((doc: any) => doc.filename);
        } else {
          this.availablePdfs = response.pdfs || [];
        }
        this.isLoadingPdfs = false;
      },
      error: (err) => {
        console.error('Failed to load PDFs:', err);
        this.availablePdfs = [];
        this.isLoadingPdfs = false;
      }
    });
  }

  onPdfSelected(pdf: string) {
    if (pdf === '') {
      this.filenameFilter = '';
      this.selectedPdf = '';
    } else {
      this.filenameFilter = pdf;
      this.selectedPdf = pdf;
    }
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

    // Save to localStorage before sending
    this.saveChatHistory();

    // Send to backend
    this.chatService.sendMessage(
      message,
      this.selectedModel,
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
        // Debug: Log the response to see what we're receiving
        console.log('Chat response received:', {
          response: response.response?.substring(0, 100),
          sources: response.sources,
          sourcesLength: response.sources?.length,
          sourcesType: typeof response.sources,
          sourcesIsArray: Array.isArray(response.sources),
          functionCalls: response.function_calls
        });
        
        // Ensure sources is always an array
        const sources = Array.isArray(response.sources) ? response.sources : (response.sources ? [response.sources] : []);
        
        console.log('Processed sources:', {
          sources,
          length: sources.length,
          firstSource: sources[0]
        });
        
        this.messages.update(msgs => [...msgs, {
          text: response.response,
          isUser: false,
          timestamp: new Date(),
          sources: sources,
          functionCalls: response.function_calls || []
        }]);
        this.isLoading.set(false);
        // Save chat history after receiving response
        this.saveChatHistory();
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
    // Clear selected PDF if manually typing
    if (value !== this.selectedPdf) {
      this.selectedPdf = '';
    }
  }

  clearFilenameFilter() {
    this.filenameFilter = '';
    this.selectedPdf = '';
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

  clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
      this.messages.set([{
        text: 'Hello! I\'m your AI assistant. How can I help you today?',
        isUser: false,
        timestamp: new Date()
      }]);
      this.expandedSources.clear();
      this.error.set(null);
      this.saveChatHistory();
    }
  }

  copyMessage(messageText: string, index: number) {
    navigator.clipboard.writeText(messageText).then(() => {
      this.copiedMessageIndex = index;
      setTimeout(() => {
        this.copiedMessageIndex = null;
      }, 2000);
    }).catch(err => {
      console.error('Failed to copy message:', err);
    });
  }

  isMessageCopied(index: number): boolean {
    return this.copiedMessageIndex === index;
  }

  private saveChatHistory() {
    try {
      const messagesToSave = this.messages().map(msg => ({
        ...msg,
        timestamp: msg.timestamp.toISOString()
      }));
      localStorage.setItem('chatHistory', JSON.stringify(messagesToSave));
      localStorage.setItem('chatSettings', JSON.stringify({
        selectedModel: this.selectedModel,
        useRag: this.useRag(),
        useFunctions: this.useFunctions(),
        filenameFilter: this.filenameFilter
      }));
    } catch (err) {
      console.error('Failed to save chat history:', err);
    }
  }

  private loadChatHistory() {
    try {
      const savedHistory = localStorage.getItem('chatHistory');
      if (savedHistory) {
        const messages = JSON.parse(savedHistory).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        this.messages.set(messages);
      }

      const savedSettings = localStorage.getItem('chatSettings');
      if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        this.selectedModel = settings.selectedModel || 'gpt-3.5-turbo';
        if (settings.useRag !== undefined) {
          this.useRag.set(settings.useRag);
        }
        if (settings.useFunctions !== undefined) {
          this.useFunctions.set(settings.useFunctions);
        }
        if (settings.filenameFilter) {
          this.filenameFilter = settings.filenameFilter;
        }
      }
    } catch (err) {
      console.error('Failed to load chat history:', err);
    }
  }
}

