import {
  Controller,
  Post,
  Body,
  BadRequestException,
} from '@nestjs/common';
import { OpenAiService } from '../openai/openai.service';

export class EmbeddingRequestDto {
  input: string | string[];
  model?: string;
}

export class EmbeddingResponseDto {
  embedding?: number[];
  embeddings?: number[][];
  model: string;
  usage?: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

@Controller('api/embeddings')
export class EmbeddingsController {
  constructor(private readonly openAiService: OpenAiService) {}

  @Post()
  async createEmbedding(
    @Body() embeddingRequest: EmbeddingRequestDto,
  ): Promise<EmbeddingResponseDto> {
    const { input, model } = embeddingRequest;

    if (!input) {
      throw new BadRequestException('Input is required');
    }

    // Validate input type
    if (typeof input === 'string' && input.trim().length === 0) {
      throw new BadRequestException('Input string cannot be empty');
    }

    if (Array.isArray(input)) {
      if (input.length === 0) {
        throw new BadRequestException('Input array cannot be empty');
      }
      // Validate all strings in array
      for (const item of input) {
        if (typeof item !== 'string' || item.trim().length === 0) {
          throw new BadRequestException(
            'All items in input array must be non-empty strings',
          );
        }
      }
    }

    const embeddingModel = model || 'text-embedding-ada-002';
    const embeddings = await this.openAiService.createEmbedding(
      input,
      embeddingModel,
    );

    // Determine if single or multiple embeddings
    const isSingle = typeof input === 'string';
    const response: EmbeddingResponseDto = {
      model: embeddingModel,
    };

    if (isSingle) {
      response.embedding = embeddings as number[];
    } else {
      response.embeddings = embeddings as number[][];
    }

    return response;
  }
}

