from django.db import models
# from django.contrib.auth.models import User
# Create your models here.
from django.db import models
# from django.contrib.auth.models import User
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
from django.conf import settings
class User(AbstractUser):
    has_donated = models.BooleanField(default=False)

    def __str__(self):
        return self.username

class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile')
    profile_picture = models.URLField(max_length=200, blank=True, null=True)  # URL to the profile picture on Backblaze
    fullname = models.CharField(max_length=255, blank=True)
    about = models.TextField(blank=True)

    @property
    def username(self):
        # Always fetch the username from the User model
        return self.user.username

    @username.setter
    def username(self, value):
        # Ensure the username is unique and valid before updating the User model
        if User.objects.filter(username=value).exclude(pk=self.user.pk).exists():
            raise ValidationError(f"Username '{value}' is already taken.")
        self.user.username = value
        self.user.save()

    @property
    def email(self):
        # Fetch the email from the User model, but don't allow updates from the profile model
        return self.user.email

    def __str__(self):
        return self.user.username


class Chat(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='chats')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Chat {self.id} for {self.user.username}"


class ChatMessage(models.Model):
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='messages')
    sender = models.CharField(max_length=10, choices=[('user', 'User'), ('bot', 'Bot')])
    message = models.TextField()
    image_url = models.URLField(blank=True, null=True)  # Store the URL of the image if provided
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender} - {self.message[:20]}..."


class Diagnosis(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='diagnoses')
    symptoms = models.TextField()  # User input symptoms
    diagnosis_result = models.TextField()  # Result from AI model
    image = models.ImageField(upload_to='diagnoses/')  # Store the image file
    image_url = models.URLField(blank=True, null=True)  # Store the Backblaze URL
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Diagnosis {self.id} for {self.user.username}"
    
class NewsSource(models.Model):
    source_id = models.CharField(max_length=255, null=True, blank=True)
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class NewsArticle(models.Model):
    source = models.ForeignKey(NewsSource, on_delete=models.CASCADE, related_name='articles',default=1)
    author = models.CharField(max_length=255, null=True, blank=True)
    title = models.CharField(max_length=255,unique=True,default='No title') 
    description = models.TextField(null=True, blank=True)
    url = models.URLField(max_length=1000)
    urlToImage = models.URLField(max_length=1000,null=True, blank=True)
    published_at = models.DateTimeField()
    content = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.title