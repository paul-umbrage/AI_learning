import { Controller, Post, Body, BadRequestException } from '@nestjs/common';
import { OpenAiService } from '../openai/openai.service';
import { DatabaseService } from '../database/database.service';

export class ChatRequestDto {
  message: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  useRag?: boolean;
  filename?: string;
  useFunctions?: boolean;
}

export class SourceDto {
  chunk_index: number;
  filename: string;
  page_number: number;
  similarity: number;
  text_preview: string;
}

export class FunctionCallDto {
  function: string;
  arguments: any;
  result: string;
}

export class ChatResponseDto {
  response: string;
  sources?: SourceDto[];
  function_calls?: FunctionCallDto[];
}

@Controller('api/chat')
export class ChatController {
  constructor(
    private readonly openAiService: OpenAiService,
    private readonly databaseService: DatabaseService,
  ) {}

  @Post()
  async chat(@Body() chatRequest: ChatRequestDto): Promise<ChatResponseDto> {
    const {
      message,
      model,
      temperature,
      maxTokens,
      useRag = true,
      filename,
      useFunctions = false,
    } = chatRequest;

    if (!message || message.trim().length === 0) {
      throw new BadRequestException('Message is required and cannot be empty');
    }

    let context = '';
    let sources: SourceDto[] = [];

    // RAG: Retrieve relevant context if enabled
    if (useRag) {
      try {
        const queryEmbedding = await this.openAiService.createEmbedding(
          message,
        );
        if (Array.isArray(queryEmbedding) && queryEmbedding.length > 0) {
          const embedding = Array.isArray(queryEmbedding[0])
            ? (queryEmbedding[0] as number[])
            : (queryEmbedding as number[]);

          const results = await this.databaseService.searchSimilarChunks(
            embedding,
            filename,
            3,
          );

          if (results.length > 0) {
            const contextParts: string[] = [];
            sources = results.map((result, index) => {
              contextParts.push(`[Context ${index + 1}]\n${result.chunk_text}\n`);
              return {
                chunk_index: index + 1,
                filename: result.filename,
                page_number: result.page_number,
                similarity: result.similarity,
                text_preview:
                  result.chunk_text.length > 100
                    ? result.chunk_text.substring(0, 100) + '...'
                    : result.chunk_text,
              };
            });
            context = contextParts.join('\n');
          }
        }
      } catch (error) {
        console.error('Error in RAG retrieval:', error);
        // Continue without RAG if there's an error
      }
    }

    // Build user message with context if RAG is enabled
    let userMessage = message;
    if (useRag && context) {
      userMessage = `Based on the following context from documents, please answer the question.

${context}

Question: ${message}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information, say so.`;
    }

    // Build system message
    let systemMessage = 'You are a helpful assistant.';
    if (useRag) {
      systemMessage +=
        ' When answering questions, use the provided context from documents. Cite sources when referencing specific information.';
    }

    const response = await this.openAiService.chatCompletionWithRAG(
      userMessage,
      systemMessage,
      model,
      temperature,
      maxTokens,
      useFunctions,
    );

    return {
      response: response.response,
      sources: sources.length > 0 ? sources : undefined,
      function_calls: response.function_calls,
    };
  }
}

