from rest_framework import serializers
# from django.contrib.auth.models import User
from rest_framework.validators import UniqueValidator
from .models import UserProfile,Chat,ChatMessage,User, Question, Answer, Comment, LikeDislike, Follower,NewsSource, NewsArticle
from django.db import transaction
from rest_framework.validators import UniqueValidator
from django.core.mail import send_mail
from django.core.exceptions import ValidationError
class UserProfileSerializer(serializers.ModelSerializer):
    # Make the email read-only
    email = serializers.ReadOnlyField()

    class Meta:
        model = UserProfile
        fields = ['profile_picture', 'username', 'fullname', 'email', 'about']

class SignupSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(
        required=True,
        validators=[UniqueValidator(queryset=User.objects.all())]
    )
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password')

    def create(self, validated_data):
        try:
            # Wrap the user creation and email sending in an atomic transaction
            with transaction.atomic():
                # Step 1: Create the user
                user = User.objects.create_user(
                    username=validated_data['username'],
                    email=validated_data['email'],
                    password=validated_data['password']
                )

                # Step 2: Send the email (with fail_silently=False to raise errors)
                subject = 'Welcome to DocTech!'
                message = f"Hi {user.username},\n\nThank you for signing up at DocTech."
                from_email = 'no-reply@doctech.com'
                recipient_list = [user.email]

                send_mail(subject, message, from_email, recipient_list, fail_silently=False)

                # Return the created user if email was sent successfully
                return user

        except Exception as e:
            # Rollback user creation and raise an error if email fails to send
            raise ValidationError("Registration failed: Could not send welcome email.") from e

        except Exception as e:
            # Catch any other unexpected exceptions
            raise ValidationError(f"Registration failed: {str(e)}")

from django.contrib.auth import get_user_model, authenticate
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers

User = get_user_model()

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def validate(cls, attrs):
        email = attrs.get('username')  # You can also use 'email' here directly if preferred
        password = attrs.get('password')
        print("Email:", email)
        print("Password:", password)

        # Retrieve user by email
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise serializers.ValidationError('Invalid email or password')

        # Authenticate the user using username and password
        user = authenticate(username=user.username, password=password)

        if user is None:
            raise serializers.ValidationError('Invalid email or password')

        # Generate the token
        token = cls.get_token(user)

        # Return token data
        return {
            'refresh': str(token),
            'access': str(token.access_token),
        }

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims to the token if needed
        token['username'] = user.username
        token['email'] = user.email
        # token['role'] = user.role  # Make sure you have a role field in your User model

        return token
    


class NewsSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsSource
        fields = ['name']

class NewsArticleSerializer(serializers.ModelSerializer):
    source = serializers.CharField(source='source.name')  # Return only the name as string

    class Meta:
        model = NewsArticle
        fields = [
            'source',
            'author',
            'title',
            'description',
            'url',
            'image_url',
            'published_at',
            'content'
        ]

class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = ['id', 'user', 'created_at']


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = '__all__'  # You can specify fields explicitly if needed

class QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Question
        fields = ['id', 'category', 'title', 'content', 'image_url', 'user', 'created_at', 'updated_at']

class AnswerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Answer
        fields = ['id', 'question', 'content', 'image_url', 'user', 'created_at', 'updated_at']


class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ['id', 'question', 'answer', 'content', 'user', 'created_at']

class LikeDislikeSerializer(serializers.ModelSerializer):
    class Meta:
        model = LikeDislike
        fields = ['id', 'question', 'answer', 'vote', 'user']

class FollowerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Follower
        fields = ['follower', 'followed', 'followed_at']