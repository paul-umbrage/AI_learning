import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { ChatController } from './chat/chat.controller';
import { EmbeddingsController } from './embeddings/embeddings.controller';
import { OpenAiService } from './openai/openai.service';
import { DatabaseService } from './database/database.service';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
  ],
  controllers: [AppController, ChatController, EmbeddingsController],
  providers: [AppService, OpenAiService, DatabaseService],
})
export class AppModule {}

