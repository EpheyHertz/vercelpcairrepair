from django.contrib import admin
from .models import UserProfile, Chat, ChatMessage, Diagnosis,User,NewsArticle,NewsSource
admin.site.register(User)
admin.site.register(NewsSource)
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'fullname', 'profile_picture')
    search_fields = ('user__username', 'fullname', 'user__email')
    list_filter = ('user__username',)

@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'created_at')
    search_fields = ('user__username',)
    list_filter = ('created_at',)

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('chat', 'sender', 'message', 'timestamp')
    search_fields = ('chat__user__username', 'message')
    list_filter = ('sender', 'timestamp')

@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'created_at', 'diagnosis_result')
    search_fields = ('user__username', 'symptoms', 'diagnosis_result')
    list_filter = ('created_at',)

# Register your models here.

# Register the NewsArticle model with the custom admin
admin.site.register(NewsArticle)