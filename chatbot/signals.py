# Create a signal receiver to handle message creation and update cache
from chatbot.models import Message
from chatbot.views import ChatbotViewSet
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

@receiver(post_save, sender=Message)
def update_chat_cache_on_message_save(sender, instance, created, **kwargs):
    chat_view = ChatbotViewSet()
    
    # Get the conversation and asset IDs
    conversation_id = instance.conversation_id
    asset_id = instance.conversation.asset_id if instance.conversation else None
    
    # Invalidate both caches
    if conversation_id:
        chat_view._invalidate_conversation_cache(conversation_id)
    if asset_id:
        chat_view._invalidate_asset_cache(asset_id)
