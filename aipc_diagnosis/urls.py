from django.urls import path
from .views import ResendVerificationCodeView, SignupView,Welcome,MyTokenObtainPairView,DiagnosisChatbotView,UserProfileView,UserChatsView,ChatMessageListCreateAPIView,PasswordResetRequestView,PasswordResetConfirmView,ContactUsView,ValidateTokenView,DeleteChatView,VerifyEmailView,LogoutView,TechNewsAPIView,TechNewsAPIViewLocalHost,VerifyCodeView
from rest_framework_simplejwt.views import TokenRefreshView
from .views import (
    QuestionAPIView,
    AnswerAPIView,
    LikeDislikeAPIView,
    CommentAPIView,
    SearchAPIView,
    FollowUserAPIView,
    UserProfileAPIView
)
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
    path('verify-code/', VerifyCodeView.as_view(), name='Mobile code veridication'),
    path('resend-verification-code/', ResendVerificationCodeView.as_view(), name='Mobile resend-verification-code'),
    path('password-reset-confirm/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('contact-us/', ContactUsView.as_view(), name='contact-us'),
    path('validate-token/', ValidateTokenView.as_view(), name='validate_token'),
    path('chats/delete/<int:chat_id>/', DeleteChatView.as_view(), name='delete_chat'),
    path('verify-email/', VerifyEmailView.as_view(), name='verify_email'),
    path('tech-news/', TechNewsAPIView.as_view(), name='tech-news'),
    path('localhosttech-news/', TechNewsAPIViewLocalHost.as_view(), name='localhosttech-news'),
     # Questions
    path('questions/', QuestionAPIView.as_view(), name='question-list-create'),  # List and create questions
    path('questions/<int:pk>/', QuestionAPIView.as_view(), name='question-detail'),  # Retrieve, update, delete a specific question

    # Answers
    path('answers/', AnswerAPIView.as_view(), name='answer-list-create'),  # List and create answers
    path('answers/<int:pk>/', AnswerAPIView.as_view(), name='answer-detail'),  # Retrieve, update, delete a specific answer

    # Like/Dislike for both questions and answers
    path('like-dislike/', LikeDislikeAPIView.as_view(), name='like-dislike'),

    # Comments on questions or answers
    path('comments/', CommentAPIView.as_view(), name='comment-list-create'),  # List and create comments

    # Search functionality for questions and answers
    path('search/', SearchAPIView.as_view(), name='search'),

    # Follow/Unfollow users
    path('follow/<int:pk>/', FollowUserAPIView.as_view(), name='follow-user'),
    path('unfollow/<int:pk>/', FollowUserAPIView.as_view(), name='unfollow-user'),
    path('users/<int:pk>/profile/', UserProfileAPIView.as_view(), name='user-profile'),
    path('logout/', LogoutView.as_view(), name='logout'),
]
