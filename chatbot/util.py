from django.core.cache import cache
def invalidate_conversation_cache(conversation_id):
    cache_key = f"chat_history_conversation_{conversation_id}"
    cache.delete(cache_key)

def invalidate_asset_cache(asset_id):
    cache_key = f"chat_history_asset_{asset_id}"
    cache.delete(cache_key)
    cache.delete(f"{cache_key}_created_at")