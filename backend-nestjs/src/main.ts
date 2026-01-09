import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Enable CORS for Angular frontend
  app.enableCors({
    origin: 'http://localhost:4200',
    credentials: true,
  });
  
  const port = process.env.PORT || 3000;
  await app.listen(port);
  console.log(`NestJS backend is running on: http://localhost:${port}`);
}
bootstrap();

