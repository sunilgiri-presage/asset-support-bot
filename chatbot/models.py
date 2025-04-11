from django.db import models
import uuid

class Conversation(models.Model):
    """Model to store chat conversations"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    asset_id = models.CharField(max_length=100, db_index=True)
    summary = models.TextField(blank=True, default="")  # New field for conversation summary
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Conversation {self.id} - Asset: {self.asset_id}"
        
class Message(models.Model):
    """Model to store individual messages within a conversation"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    is_user = models.BooleanField(default=True, help_text="True if message is from user, False if from system")
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{'User' if self.is_user else 'System'} message in {self.conversation.id}"
