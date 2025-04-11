# chatbot/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ChatbotViewSet

router = DefaultRouter()
# router.register(r'conversations', ConversationViewSet)
router.register(r'', ChatbotViewSet, basename='chatbot')

urlpatterns = [
    path('', include(router.urls))
]