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
    class Meta:
        model = Conversation
        fields = ['id', 'asset_id', 'summary', 'created_at', 'updated_at']

class MessagePairSerializer(serializers.Serializer):
    conversation = ConversationSerializer()
    user_message = MessageSerializer(allow_null=True)
    system_message = MessageSerializer(allow_null=True)

class QuerySerializer(serializers.Serializer):
    asset_id = serializers.CharField(required=True)
    message = serializers.CharField(required=True)
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
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
    asset_type = serializers.CharField(allow_blank=True, default="Unknown")
    running_RPM = serializers.FloatField(default=0)
    bearing_fault_frequencies = serializers.DictField(default=dict)
    acceleration_time_waveform = serializers.DictField(default=dict)
    velocity_time_waveform = serializers.DictField(default=dict)
    harmonics = serializers.DictField(default=dict)
    cross_PSD = serializers.ListField(default=list)
