# NestJS Backend for AI Learning Project

This is a NestJS backend that provides an API to call the OpenAI API.

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- OpenAI API key

## Setup Instructions

1. Navigate to the backend-nestjs directory:
   ```bash
   cd backend-nestjs
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the backend-nestjs directory:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PORT=3000
   ```

5. Run the backend server:
   ```bash
   # Development mode (with hot reload)
   npm run start:dev

   # Production mode
   npm run build
   npm run start:prod
   ```

   The backend will be available at `http://localhost:3000` (or the port specified in your .env file)

## API Endpoints

### Root Endpoint
- `GET /` - Returns a welcome message
- `GET /health` - Health check endpoint

### Chat Endpoint
- `POST /api/chat` - Send a message to OpenAI
  - Request body:
    ```json
    {
      "message": "Your message here",
      "model": "gpt-3.5-turbo",  // optional, defaults to gpt-3.5-turbo
      "temperature": 0.7,         // optional, defaults to 0.7
      "maxTokens": 500            // optional, defaults to 500
    }
    ```
  - Response:
    ```json
    {
      "response": "AI response here"
    }
    ```

### Embeddings Endpoint
- `POST /api/embeddings` - Generate embeddings for text using OpenAI API
  - Request body (single text):
    ```json
    {
      "input": "Your text here",
      "model": "text-embedding-ada-002"  // optional, defaults to text-embedding-ada-002
    }
    ```
  - Request body (multiple texts):
    ```json
    {
      "input": ["Text 1", "Text 2", "Text 3"],
      "model": "text-embedding-ada-002"  // optional
    }
    ```
  - Response (single input):
    ```json
    {
      "embedding": [0.123, 0.456, ...],
      "model": "text-embedding-ada-002"
    }
    ```
  - Response (multiple inputs):
    ```json
    {
      "embeddings": [[0.123, 0.456, ...], [0.789, 0.012, ...]],
      "model": "text-embedding-ada-002"
    }
    ```

## Project Structure

```
backend-nestjs/
├── src/
│   ├── app.controller.ts      # Root controller
│   ├── app.module.ts          # Main application module
│   ├── app.service.ts         # Root service
│   ├── main.ts                # Application entry point
│   ├── chat/
│   │   └── chat.controller.ts    # Chat API controller
│   ├── embeddings/
│   │   └── embeddings.controller.ts # Embeddings API controller
│   └── openai/
│       └── openai.service.ts     # OpenAI service (chat & embeddings)
├── .env.example               # Environment variables template
├── package.json               # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
└── nest-cli.json              # NestJS CLI configuration
```

## Development

- The backend uses NestJS with TypeScript
- Hot reload is enabled in development mode (`npm run start:dev`)
- CORS is enabled for `http://localhost:4200` (Angular default port)

## Troubleshooting

- Make sure your OpenAI API key is correctly set in the `.env` file
- Check that port 3000 (or your specified port) is not already in use
- Verify all npm dependencies are installed with `npm install`
- Check console logs for any error messages

