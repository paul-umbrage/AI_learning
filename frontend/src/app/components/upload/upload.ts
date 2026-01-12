import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { HttpClient, HttpEventType, HttpResponse } from '@angular/common/http';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './upload.html',
  styleUrl: './upload.css'
})
export class UploadComponent {
  selectedFile: File | null = null;
  isUploading = false;
  uploadProgress = 0;
  uploadStatus: 'idle' | 'success' | 'error' = 'idle';
  uploadMessage = '';
  showConfirmation = false;
  uploadResult: any = null;

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      if (file.type === 'application/pdf') {
        this.selectedFile = file;
        this.uploadStatus = 'idle';
        this.uploadMessage = '';
      } else {
        this.uploadMessage = 'Please select a PDF file';
        this.uploadStatus = 'error';
        this.selectedFile = null;
      }
    }
  }

  uploadFile() {
    if (!this.selectedFile) {
      this.uploadMessage = 'Please select a file first';
      this.uploadStatus = 'error';
      return;
    }

    this.isUploading = true;
    this.uploadProgress = 10;
    this.uploadStatus = 'idle';
    this.uploadMessage = 'Uploading and processing PDF (this may take a few minutes)...';

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    // Use a simple POST request with a long timeout
    // Processing happens server-side, so we just wait for the response
    this.http.post<any>('http://localhost:8000/api/upload', formData, {
      // Set timeout to 10 minutes for large PDFs
    }).subscribe({
      next: (response) => {
        console.log('Upload response received:', response);
        this.handleUploadSuccess(response);
      },
      error: (error) => {
        console.error('Upload error:', error);
        this.handleUploadError(error);
      }
    });

    // Simulate progress since we can't track server-side processing
    // This gives user feedback that something is happening
    let progress = 10;
    const progressInterval = setInterval(() => {
      if (!this.isUploading) {
        clearInterval(progressInterval);
        return;
      }
      progress = Math.min(progress + 2, 90); // Gradually increase to 90%
      this.uploadProgress = progress;
    }, 2000); // Update every 2 seconds

    // Clean up interval when upload completes
    setTimeout(() => {
      clearInterval(progressInterval);
    }, 600000); // 10 minutes max
  }

  closeConfirmation() {
    this.showConfirmation = false;
    this.selectedFile = null;
    this.uploadStatus = 'idle';
    this.uploadMessage = '';
    this.uploadResult = null;
    this.uploadProgress = 0;
    
    // Reset file input
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  }

  removeFile() {
    this.selectedFile = null;
    this.uploadStatus = 'idle';
    this.uploadMessage = '';
    
    // Reset file input
    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  }

  private handleUploadSuccess(result: any) {
    console.log('Upload success:', result);
    this.uploadResult = result;
    this.uploadStatus = 'success';
    this.uploadMessage = `Successfully uploaded and ingested ${result?.filename || 'PDF'}`;
    this.showConfirmation = true;
    this.isUploading = false;
    this.uploadProgress = 100;
  }

  private handleUploadError(error: any) {
    console.error('Upload error:', error);
    this.isUploading = false;
    this.uploadStatus = 'error';
    
    // Provide more detailed error messages
    if (error.status === 0) {
      this.uploadMessage = 'Connection error. Make sure the backend server is running on port 8000.';
    } else if (error.status === 500) {
      this.uploadMessage = error.error?.detail || 'Server error during processing. Check backend logs.';
    } else if (error.status === 400) {
      this.uploadMessage = error.error?.detail || 'Invalid file or processing failed.';
    } else if (error.status === 504 || error.name === 'TimeoutError') {
      this.uploadMessage = 'Request timed out. The PDF might be too large. Try a smaller file.';
    } else {
      this.uploadMessage = error.error?.detail || error.message || 'Failed to upload PDF';
    }
    
    this.uploadProgress = 0;
  }
}
