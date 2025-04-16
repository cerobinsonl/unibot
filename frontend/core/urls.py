from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('data_analysis/', views.data_analysis_view, name='data_analysis'),
    path('send_messages/', views.send_messages_view, name='send_messages'),
    path('input_data/', views.input_data_view, name='input_data'),
    path('extract_data/', views.extract_data_view, name='extract_data'),
    path('create_synthetic_data/', views.create_synthetic_data_view, name='create_synthetic_data'),
    
    # Add chatbot API endpoint
    path('api/chatbot/message', views.chatbot_message, name='chatbot_message'),
]