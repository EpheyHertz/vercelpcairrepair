from django.urls import path
from .views import SignupView,Welcome,MyTokenObtainPairView,DiagnosisChatbotView,UserProfileView,UserChatsView,ChatMessageListCreateAPIView,PasswordResetRequestView,PasswordResetConfirmView,ContactUsView,ValidateTokenView,DeleteChatView,VerifyEmailView,LogoutView,TechNewsAPIView
from rest_framework_simplejwt.views import TokenRefreshView
urlpatterns = [
    path('', Welcome.as_view(), name='welcome'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('chatbot-diagnose/', DiagnosisChatbotView.as_view(), name='chatbot_diagnose'),
    path('profile/', UserProfileView.as_view(), name='user-profile'),
    path('user/chats/', UserChatsView.as_view(), name='user-chats'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('chats/<int:pk>/messages/', ChatMessageListCreateAPIView.as_view(), name='chat-message-list-create'),
    path('password-reset/', PasswordResetRequestView.as_view(), name='password_reset'),
    path('password-reset-confirm/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('contact-us/', ContactUsView.as_view(), name='contact-us'),
    path('validate-token/', ValidateTokenView.as_view(), name='validate_token'),
    path('chats/delete/<int:chat_id>/', DeleteChatView.as_view(), name='delete_chat'),
    path('verify-email/', VerifyEmailView.as_view(), name='verify_email'),
    path('tech-news/', TechNewsAPIView.as_view(), name='tech-news'),
    path('logout/', LogoutView.as_view(), name='logout'),
]
