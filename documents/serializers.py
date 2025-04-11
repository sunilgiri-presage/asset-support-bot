from rest_framework import serializers
from .models import Document
from django.conf import settings

class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for Document model with enhanced file validation"""
    class Meta:
        model = Document
        fields = [
            'id', 'asset_id', 'title', 'file', 
            'content_type', 'file_size', 'is_processed', 
            'processing_error', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'is_processed', 'processing_error', 
            'created_at', 'updated_at', 'file_size', 'content_type'
        ]

    def validate_file(self, value):
        """Validate file size and type"""
        # Check file size (adjust as needed)
        max_upload_size = getattr(settings, 'MAX_UPLOAD_SIZE', 10 * 1024 * 1024)  # Default 10 MB
        if value.size > max_upload_size:
            raise serializers.ValidationError(
                f"File size exceeds the limit of {max_upload_size / (1024 * 1024)} MB"
            )

        # Check file type (optional)
        allowed_types = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain'
        ]
        
        # Try to get content type from different attributes
        content_type = getattr(value, 'content_type', None)
        if not content_type and hasattr(value, 'file'):
            content_type = getattr(value.file, 'content_type', None)
        
        if content_type not in allowed_types:
            raise serializers.ValidationError(
                f"Unsupported file type: {content_type}. Please upload PDF, DOCX, DOC, or TXT files."
            )

        return value

    def create(self, validated_data):
        """Handle file upload data with robust content type detection"""
        file = validated_data.get('file')
        
        # Try to get content type from different sources
        if file:
            content_type = getattr(file, 'content_type', None)
            if not content_type and hasattr(file, 'file'):
                content_type = getattr(file.file, 'content_type', None)
            
            if content_type:
                validated_data['content_type'] = content_type
            
            validated_data['file_size'] = file.size
        
        return super().create(validated_data)