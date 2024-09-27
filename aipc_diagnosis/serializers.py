from rest_framework import serializers
# from django.contrib.auth.models import User
from rest_framework.validators import UniqueValidator
from .models import UserProfile,Chat,ChatMessage,User

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
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password']
        )
        return user



from django.contrib.auth import get_user_model, authenticate
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers

User = get_user_model()

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def validate(cls, attrs):
        email = attrs.get('username')  # You can also use 'email' here directly if preferred
        password = attrs.get('password')

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
class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = ['id', 'user', 'created_at']


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = '__all__'  # You can specify fields explicitly if needed