import { Routes } from '@angular/router';
import { ChatComponent } from './components/chat/chat.component';
import { UploadComponent } from './components/upload/upload';

export const routes: Routes = [
  { path: '', component: ChatComponent },
  { path: 'upload', component: UploadComponent },
  { path: '**', redirectTo: '' }
];
