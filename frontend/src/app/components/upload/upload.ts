import { Component, ChangeDetectorRef } from '@angular/core';
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
  private progressInterval: any = null;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef
  ) {}

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

    // Clear any existing progress interval
    this.clearProgressInterval();

    // Reset state
    this.isUploading = true;
    this.uploadProgress = 10;
    this.uploadStatus = 'idle';
    this.uploadMessage = 'Uploading and processing PDF (this may take a few minutes)...';
    this.uploadResult = null;
    this.showConfirmation = false;

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    console.log('Starting upload for file:', this.selectedFile.name);

    // Make POST request to upload endpoint
    this.http.post<any>('http://localhost:8000/api/upload', formData, {
      // No special options needed - default behavior should work
    }).subscribe({
      next: (response) => {
        console.log('✅ Upload response received:', response);
        console.log('Response type:', typeof response);
        console.log('Response data:', JSON.stringify(response));
        
        // Handle the response - it should be the data directly from FastAPI
        const result = response;
        this.handleUploadSuccess(result);
      },
      error: (error) => {
        console.error('❌ Upload error:', error);
        console.error('Error status:', error?.status);
        console.error('Error message:', error?.message);
        console.error('Error body:', error?.error);
        this.handleUploadError(error);
      },
      complete: () => {
        console.log('✅ Upload request observable completed');
      }
    });

    // Simulate progress since we can't track server-side processing
    // This gives user feedback that something is happening
    let progress = 10;
    this.progressInterval = setInterval(() => {
      // Check if upload is still in progress
      if (!this.isUploading) {
        console.log('Upload no longer in progress, clearing interval');
        this.clearProgressInterval();
        return;
      }
      // Only update if we haven't reached completion
      if (this.uploadProgress < 95) {
        progress = Math.min(progress + 2, 95); // Gradually increase to 95% (leave room for completion)
        this.uploadProgress = progress;
        console.log('Progress updated to:', this.uploadProgress + '%');
      }
    }, 2000); // Update every 2 seconds
  }

  private clearProgressInterval() {
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
      this.progressInterval = null;
    }
  }

  closeConfirmation() {
    this.showConfirmation = false;
    this.selectedFile = null;
    this.uploadStatus = 'idle';
    this.uploadMessage = '';
    this.uploadResult = null;
    this.uploadProgress = 0;
    this.clearProgressInterval();
    
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
    console.log('🎉 Handling upload success:', result);
    
    // Clear progress interval FIRST to prevent any race conditions
    this.clearProgressInterval();
    console.log('Progress interval cleared');
    
    // Immediately update progress to 100% before changing other state
    this.uploadProgress = 100;
    console.log('Progress set to 100%');
    
    // Update state immediately
    this.uploadResult = result;
    this.uploadStatus = 'success';
    this.uploadMessage = `Successfully uploaded and ingested ${result?.filename || 'PDF'}`;
    this.showConfirmation = true;
    this.isUploading = false;
    
    // Force change detection
    this.cdr.detectChanges();
    
    console.log('✅ Upload state updated successfully');
    console.log('Upload result:', this.uploadResult);
    console.log('Upload status:', this.uploadStatus);
    console.log('Show confirmation:', this.showConfirmation);
    console.log('Is uploading:', this.isUploading);
    console.log('Upload progress:', this.uploadProgress);
  }

  private handleUploadError(error: any) {
    console.error('❌ Handling upload error:', error);
    // Clear progress interval first
    this.clearProgressInterval();
    
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
    
    // Force change detection
    this.cdr.detectChanges();
  }
}
