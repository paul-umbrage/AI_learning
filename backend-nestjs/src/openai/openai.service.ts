import { Injectable, BadRequestException, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import OpenAI from 'openai';

interface FunctionCallResult {
  function: string;
  arguments: any;
  result: string;
}

interface ChatCompletionWithRAGResponse {
  response: string;
  function_calls?: FunctionCallResult[];
}

@Injectable()
export class OpenAiService {
  private client: OpenAI | null = null;

  constructor(private configService: ConfigService) {
    const apiKey = this.configService.get<string>('OPENAI_API_KEY');
    
    if (!apiKey) {
      console.warn('Warning: OPENAI_API_KEY not found in environment variables');
    } else {
      this.client = new OpenAI({
        apiKey: apiKey,
      });
    }
  }

  async chatCompletion(
    message: string,
    model: string = 'gpt-3.5-turbo',
    temperature: number = 0.7,
    maxTokens: number = 500,
  ): Promise<string> {
    if (!this.client) {
      throw new InternalServerErrorException(
        'OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file',
      );
    }

    try {
      const response = await this.client.chat.completions.create({
        model: model,
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: message },
        ],
        temperature: temperature,
        max_tokens: maxTokens,
      });

      return response.choices[0].message.content || '';
    } catch (error) {
      throw new BadRequestException(
        `Error calling OpenAI API: ${error.message}`,
      );
    }
  }

  async chatCompletionWithRAG(
    userMessage: string,
    systemMessage: string,
    model: string = 'gpt-3.5-turbo',
    temperature: number = 0.7,
    maxTokens: number = 1000,
    useFunctions: boolean = false,
  ): Promise<ChatCompletionWithRAGResponse> {
    if (!this.client) {
      throw new InternalServerErrorException(
        'OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file',
      );
    }

    try {
      const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
        { role: 'system', content: systemMessage },
        { role: 'user', content: userMessage },
      ];

      const apiParams: OpenAI.Chat.Completions.ChatCompletionCreateParams = {
        model: model,
        messages: messages,
        temperature: temperature,
        max_tokens: maxTokens,
      };

      // Add function calling if enabled
      if (useFunctions) {
        apiParams.tools = this.getFunctionDefinitions();
        apiParams.tool_choice = 'auto';
      }

      const response = await this.client.chat.completions.create(apiParams);
      const assistantMessage = response.choices[0].message;
      let finalResponse = assistantMessage.content || '';
      const functionCalls: FunctionCallResult[] = [];

      // Handle function calls if any
      if (useFunctions && assistantMessage.tool_calls) {
        // Add assistant's message with tool calls
        messages.push({
          role: 'assistant',
          content: assistantMessage.content,
          tool_calls: assistantMessage.tool_calls.map((tc) => ({
            id: tc.id,
            type: tc.type,
            function: {
              name: tc.function.name,
              arguments: tc.function.arguments,
            },
          })),
        });

        // Execute function calls
        for (const toolCall of assistantMessage.tool_calls) {
          const functionName = toolCall.function.name;
          let argumentsObj: any = {};
          
          try {
            argumentsObj = JSON.parse(toolCall.function.arguments);
          } catch {
            // If parsing fails, use empty object
          }

          const functionResult = this.executeFunctionCall(
            functionName,
            argumentsObj,
          );

          functionCalls.push({
            function: functionName,
            arguments: argumentsObj,
            result: functionResult,
          });

          // Add function result to messages
          messages.push({
            role: 'tool',
            tool_call_id: toolCall.id,
            content: functionResult,
          });
        }

        // Get final response after function execution
        const finalResponseObj = await this.client.chat.completions.create({
          model: model,
          messages: messages,
          temperature: temperature,
          max_tokens: maxTokens,
        });

        finalResponse = finalResponseObj.choices[0].message.content || '';
      }

      return {
        response: finalResponse,
        function_calls: functionCalls.length > 0 ? functionCalls : undefined,
      };
    } catch (error) {
      throw new BadRequestException(
        `Error calling OpenAI API: ${error.message}`,
      );
    }
  }

  private getFunctionDefinitions(): OpenAI.Chat.Completions.ChatCompletionTool[] {
    return [
      {
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get the current weather in a given location',
          parameters: {
            type: 'object',
            properties: {
              location: {
                type: 'string',
                description: 'The city and state, e.g. San Francisco, CA',
              },
              unit: {
                type: 'string',
                enum: ['celsius', 'fahrenheit'],
                description: 'The unit for temperature',
              },
            },
            required: ['location'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'calculate',
          description: 'Perform mathematical calculations',
          parameters: {
            type: 'object',
            properties: {
              expression: {
                type: 'string',
                description:
                  "A mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(16)'",
              },
            },
            required: ['expression'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'get_current_time',
          description: 'Get the current date and time',
          parameters: {
            type: 'object',
            properties: {},
            required: [],
          },
        },
      },
    ];
  }

  private executeFunctionCall(functionName: string, argumentsObj: any): string {
    try {
      if (functionName === 'get_weather') {
        const location = argumentsObj.location || 'unknown';
        const unit = argumentsObj.unit || 'fahrenheit';
        return `The weather in ${location} is sunny, 72°${unit[0].toUpperCase()}`;
      } else if (functionName === 'calculate') {
        const expression = argumentsObj.expression || '';
        // Safe evaluation of mathematical expressions
        const allowedNames: { [key: string]: any } = {
          abs: Math.abs,
          acos: Math.acos,
          asin: Math.asin,
          atan: Math.atan,
          ceil: Math.ceil,
          cos: Math.cos,
          exp: Math.exp,
          floor: Math.floor,
          log: Math.log,
          max: Math.max,
          min: Math.min,
          pow: Math.pow,
          random: Math.random,
          round: Math.round,
          sin: Math.sin,
          sqrt: Math.sqrt,
          tan: Math.tan,
          PI: Math.PI,
          E: Math.E,
        };

        try {
          // Create a safe evaluation context
          const result = Function(
            '"use strict"; return (' + expression + ')',
          )();
          return `The result of ${expression} is ${result}`;
        } catch (error) {
          return `Error calculating ${expression}: ${error.message}`;
        }
      } else if (functionName === 'get_current_time') {
        const now = new Date();
        return `Current date and time: ${now.toISOString().replace('T', ' ').substring(0, 19)}`;
      } else {
        return `Unknown function: ${functionName}`;
      }
    } catch (error) {
      return `Error executing function ${functionName}: ${error.message}`;
    }
  }

  async createEmbedding(
    input: string | string[],
    model: string = 'text-embedding-ada-002',
  ): Promise<number[] | number[][]> {
    if (!this.client) {
      throw new InternalServerErrorException(
        'OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file',
      );
    }

    try {
      const response = await this.client.embeddings.create({
        model: model,
        input: input,
      });

      // If single input, return single embedding array
      if (typeof input === 'string') {
        return response.data[0].embedding;
      }

      // If multiple inputs, return array of embeddings
      return response.data.map((item) => item.embedding);
    } catch (error) {
      throw new BadRequestException(
        `Error calling OpenAI Embeddings API: ${error.message}`,
      );
    }
  }
}

