# chatbot/serializers.py
from rest_framework import serializers
from .models import Conversation, Message

class MessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages"""
    class Meta:
        model = Message
        fields = ['id', 'is_user', 'content', 'created_at']
        read_only_fields = ['id', 'created_at']

class ConversationSerializer(serializers.ModelSerializer):
    """Original serializer for conversations"""
    messages = MessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Conversation
        fields = ['id', 'asset_id', 'messages', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

class QuerySerializer(serializers.Serializer):
    """Serializer for chat queries"""
    asset_id = serializers.CharField(required=True)
    message = serializers.CharField(required=True)
    conversation_id = serializers.UUIDField(required=False)
    use_search = serializers.BooleanField(required=False, default=False)

class MessagePairSerializer(serializers.Serializer):
    """Serializer for a pair of user message and system response"""
    id = serializers.UUIDField(source='conversation.id')
    asset_id = serializers.CharField(source='conversation.asset_id')
    messages = serializers.SerializerMethodField()
    created_at = serializers.DateTimeField(source='conversation.created_at')
    updated_at = serializers.DateTimeField(source='conversation.updated_at')

    def get_messages(self, obj):
        user_message = obj.get('user_message')
        system_message = obj.get('system_message')
        
        messages = []
        if user_message:
            messages.append({
                'id': str(user_message.id),
                'is_user': user_message.is_user,
                'content': user_message.content,
                'created_at': user_message.created_at
            })
        if system_message:
            messages.append({
                'id': str(system_message.id),
                'is_user': system_message.is_user,
                'content': system_message.content,
                'created_at': system_message.created_at
            })
        
        return messages
    
class VibrationAnalysisInputSerializer(serializers.Serializer):
    asset_type = serializers.CharField()
    running_RPM = serializers.FloatField()
    bearing_fault_frequencies = serializers.DictField(child=serializers.FloatField())
    acceleration_time_waveform = serializers.DictField()
    velocity_time_waveform = serializers.DictField()
    harmonics = serializers.DictField()
    cross_PSD = serializers.DictField()