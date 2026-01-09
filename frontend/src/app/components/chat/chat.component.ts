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
  selectedFilename = signal<string | undefined>(undefined);

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
      this.selectedFilename(),
      this.useFunctions()
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
}

