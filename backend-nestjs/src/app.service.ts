import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  getHello(): string {
    return 'AI Learning NestJS Backend API is running';
  }

  getHealth(): object {
    return { status: 'healthy' };
  }
}

