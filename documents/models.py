from django.db import models
import uuid
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile

class DatabaseStorage(FileSystemStorage):
    """
    Custom storage backend to handle file uploads more robustly
    """
    def __init__(self, *args, **kwargs):
        kwargs['location'] = settings.MEDIA_ROOT
        super().__init__(*args, **kwargs)

class Document(models.Model):
    """Model to store document metadata and file content"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    asset_id = models.CharField(max_length=100, db_index=True, help_text="Unique ID of the asset this document belongs to")
    title = models.CharField(max_length=255)
    
    file = models.FileField(
        upload_to='documents/',
        storage=DatabaseStorage(),
        help_text="Document file stored in the database"
    )
    
    content_type = models.CharField(max_length=100, blank=True, null=True)
    file_size = models.IntegerField(default=0)
    is_processed = models.BooleanField(default=False, help_text="Whether document has been processed for embeddings")
    processing_error = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} - Asset: {self.asset_id}"

    def save(self, *args, **kwargs):
        # Ensure file size is set
        if self.file:
            # Check if the file is an UploadedFile
            if hasattr(self.file, 'content_type'):
                self.content_type = self.file.content_type
            elif hasattr(self.file, 'file') and hasattr(self.file.file, 'content_type'):
                self.content_type = self.file.file.content_type
            
            # Set file size
            try:
                self.file_size = self.file.size
            except Exception:
                # Fallback if size cannot be determined
                self.file_size = 0
        
        super().save(*args, **kwargs)