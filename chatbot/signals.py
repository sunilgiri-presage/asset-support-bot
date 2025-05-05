from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import Message
from .util import invalidate_conversation_cache, invalidate_asset_cache

@receiver(post_save, sender=Message)
def update_chat_cache_on_message_save(sender, instance, **kwargs):
    invalidate_conversation_cache(instance.conversation_id)
    invalidate_asset_cache(instance.conversation.asset_id)

@receiver(post_delete, sender=Message)
def update_chat_cache_on_message_delete(sender, instance, **kwargs):
    invalidate_conversation_cache(instance.conversation_id)
    invalidate_asset_cache(instance.conversation.asset_id)