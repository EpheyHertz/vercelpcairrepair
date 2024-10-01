from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
from .serializers import SignupSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import MyTokenObtainPairSerializer,ChatSerializer,ChatMessageSerializer
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework import generics
from PIL import Image
from rest_framework.permissions import AllowAny
from django.core.mail import send_mail
from django.contrib.auth.tokens import default_token_generator
# from django.contrib.auth.models import User
from .models import UserProfile,User
import http.client
import urllib.parse
import json
from .serializers import UserProfileSerializer
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.db import IntegrityError
from newsapi import NewsApiClient
import io
import requests
import os
import tempfile
from django.contrib.auth import logout
import smtplib
from .models import Chat, ChatMessage, Diagnosis
from rest_framework.exceptions import ValidationError
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
# from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from django.contrib.auth.tokens import default_token_generator


import google.generativeai as genai
GEMINI_AI_API_KEY=settings.GEMINI_AI_API_KEY
genai.configure(api_key=GEMINI_AI_API_KEY)
from .utils import upload_image_to_backblaze  # Import your image upload utility function
import logging

logger = logging.getLogger(__name__)
class Welcome(APIView):
    permission_classes = [AllowAny]
    def get(self,request):
        return Response({"message":"Welcome to pc diagnosis apis"})
    

# class SignupView(APIView):
#     def post(self, request):
#         # Step 1: Use the SignupSerializer for validation
#         serializer = SignupSerializer(data=request.data)

#         if serializer.is_valid():
#             # Step 2: Create the user using validated data
#             try:
#                 user = serializer.save()

#                 # Step 3: Attempt to send the welcome email
#                 try:
#                     send_mail(
#                         'Welcome to DocTech - Your Repair Companion',
#                         f'''
#                         Dear {user.username},

#                         Welcome to DocTech! We are excited to have you on board as a member of our community dedicated to the repair of PCs, tablets, phones, and laptops.

#                         As a member of DocTech, you now have access to a range of tools and resources designed to help you diagnose and repair a variety of tech issues. Whether you're experiencing software glitches, hardware failures, or need general tech support, our platform is here to assist you.

#                         If you have any questions or need assistance, feel free to reach out to our support team at epheynyaga@gmail.com.

#                         We're thrilled to have you with us and look forward to supporting you on your repair journey!

#                         Best regards,
#                         The DocTech Team
#                         ''',
#                         settings.DEFAULT_FROM_EMAIL,
#                         [user.email],
#                         fail_silently=False,  # Fail loudly to catch any SMTP errors
#                     )
#                 except Exception as e:
#                     # Rollback user creation if email fails
#                     user.delete()  # Remove the user if the email wasn't sent successfully
#                     return Response({'error': f'Failed to send welcome email: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#                 # Step 4: Return success response if everything works
#                 return Response({'message': 'User registered successfully!'}, status=status.HTTP_201_CREATED)

#             except Exception as e:
#                 return Response({'error': f'Error during registration: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#         # If serializer is not valid, return the errors
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# from django.urls import reverse
# from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
# from django.utils.encoding import force_bytes
# from .utils import email_verification_token  # Import the token generator
# from django.contrib.sites.shortcuts import get_current_site



# class SignupView(APIView):
#     def post(self, request):
#         username = request.data.get('username')
#         email = request.data.get('email')
#         password = request.data.get('password')

#         # Check for required fields
#         if not username or not email or not password:
#             return Response({'error': 'Username, email, and password are required'}, status=status.HTTP_400_BAD_REQUEST)

#         # Check if username is already taken
#         if User.objects.filter(username=username).exists():
#             return Response({'error': 'Username is already taken'}, status=status.HTTP_400_BAD_REQUEST)

#         # Check if email is already registered
#         if User.objects.filter(email=email).exists():
#             return Response({'error': 'Email is already registered'}, status=status.HTTP_400_BAD_REQUEST)
#         user = User(username=username, email=email)
#         user.set_password(password)
#         try:
#             try:
#                 send_mail(
                    
#                     'Welcome to DocTech - Your Repair Companion',
#                     f'''
#                     Dear {username},

#                     Welcome to DocTech! We are excited to have you on board as a member of our community dedicated to the repair of PCs, tablets, phones, and laptops.

#                     As a member of DocTech, you now have access to a range of tools and resources designed to help you diagnose and repair a variety of tech issues. Whether you're experiencing software glitches, hardware failures, or need general tech support, our platform is here to assist you.

#                     If you have any questions or need assistance, feel free to reach out to our support team at epheynyaga@gmail.com.

#                     We're thrilled to have you with us and look forward to supporting you on your repair journey!

#                     Best regards,
#                     The DocTech Team
#                     ''',
#                     settings.DEFAULT_FROM_EMAIL,
#                     [email],
#                     fail_silently=False,  # Don't fail silently to catch errors
            

#                 )
#             except Exception as e:
#                 return Response({'error': f'Failed to send welcome email: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#             # If email is sent successfully, save the user
            
#             user.save()

#             return Response({'message': 'User registered successfully!'}, status=status.HTTP_201_CREATED)

#         except Exception as e:
#             # If email sending fails or any other error occurs, return an error response
#             return Response({'error': f'Failed to send welcome email: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#         except IntegrityError:
#             return Response({'error': 'An error occurred while creating the user. Please try again.'}, status=status.HTTP_400_BAD_REQUEST)

#         except ValidationError as e:
#             return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


class ValidateTokenView(APIView):
    # authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            # Since the token is checked in the background by DRF's authentication system,
            # we simply return a success response if the token is valid
            return Response({"error": "Token is valid"}, status=200)
        except TokenError as e:
            # If the token is invalid or expired
            return Response({"detail": str(e)}, status=401)

class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        """
        Retrieve the user's profile or create one if it doesn't exist.
        """
        user = self.request.user
        if not hasattr(user, 'profile'):
            # Create a profile if it doesn't exist
            profile = UserProfile.objects.create(user=user)
        else:
            profile = user.profile
        return profile

    def perform_update(self, serializer):
        profile = self.get_object()

        # Handle image upload if an image is provided
        if self.request.FILES.get('picture'):
            picture = self.request.FILES['picture']
            # Upload the image to Backblaze
            backblaze_url =upload_image_to_backblaze(picture)

            if backblaze_url:
                # Update the profile picture URL
                profile.profile_picture = backblaze_url

        # Save the other profile fields
        serializer.save()
class DeleteChatView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, chat_id):
        # Get the chat object or return 404 if it doesn't exist
        chat = get_object_or_404(Chat, id=chat_id, user=request.user)
        
        # Delete the chat
        chat.delete()
        
        return Response({'detail': 'Chat deleted successfully.'}, status=status.HTTP_204_NO_CONTENT)

# class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
#     @classmethod
#     def validate(cls, attrs):
#         email = attrs.get('username')
#         password = attrs.get('password')

#         # Retrieve user by email
#         try:
#             user = User.objects.get(email=email)
#         except User.DoesNotExist:
#             raise serializers.ValidationError('Invalid email or password')

#         # Authenticate the user using username and password
#         user = authenticate(username=user.username, password=password)

#         if user is None:
#             raise serializers.ValidationError('Invalid email or password')

#         # Get the token using the parent class method
#         token = cls.get_token(user)

#         # Return token data
#         return {
#             'refresh': str(token),
#             'access': str(token.access_token),
#         }

#     @classmethod
#     def get_token(cls, user):
#         token = super().get_token(user)

#         # Add custom claims
#         token['username'] = user.username
#         token['email'] = user.email
#         # token['role'] = user.role  # Make sure you have a role field in your User model

#         return token
class ChatMessageListCreateAPIView(generics.ListCreateAPIView):
    serializer_class = ChatMessageSerializer
    permission_classes = [IsAuthenticated]  # Adjust permissions as needed

    def get_queryset(self):
        """
        Restrict the returned chat messages to the specified chat
        identified by `pk` in the URL.
        """
        chat_id = self.kwargs.get('pk')  # Get the `pk` from the URL
        if chat_id is not None:
            return ChatMessage.objects.filter(chat_id=chat_id)
        return ChatMessage.objects.none()  # Return an empty queryset if no chat_id is found


generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction = (
    "DocTech Technical Support Chatbot Prompt:\n\n"
    "Welcome to DocTech , your technical assistant powered by Gemini. "
    "When a user asks, Who made you? respond with the following message:"
    "I was developed by Ephey Nyaga, a Computer Science student at Embu University in Kenya."
    "Please note that I am specifically designed to assist with technical issues related to PCs, laptops, phones, and tablets. "
    "You are an AI diagnostic assistant that specializes in analyzing images of PCs, "
    "laptops, and phones. Your task is to identify any visible hardware or software "
    "issues based on the provided image. Consider aspects such as physical damage, "
    "screen quality, and any indicators of software failure. Provide a detailed diagnosis, "
    "including potential problems and recommendations for resolution."
    "If you provide questions or concerns outside this scope, I will kindly remind you that my expertise lies in technology support.\n"
    "I am here to help you diagnose and resolve minor technical problems and provide guidance based on your descriptions. "
    "Before asking for details, I will review the chat history to gather any available user information. If I cannot find relevant details, I will ask you for the following:\n\n"
    "* **Full Name:**\n"
    "* **Email Address:**\n"
    "* **Device Type:** (PC, Laptop, Phone, Tablet)\n"
    "* **Issue Description:**\n\n"
    "If the user uploads an image and the image url is provided,please analyze the provided image for any anomalies or issues that may impact quality, usability, or compliance with industry standards. Specifically, assess the image for visual clarity, presence of any inappropriate content, quality degradation (such as blurriness, pixelation, or distortion), and ensure that it adheres to the guidelines for acceptable media. The diagnostic should also evaluate whether the image is suitable for use in professional environments, flagging any content that could be deemed offensive or irrelevant to the context of use. Provide detailed feedback and, if possible, suggest ways to enhance the image quality or rectify identified issues."
    "Once I have this information, I will perform a thorough investigation of your technical issues using trusted technical resources. "
    "Feel free to ask more questions to gather concrete information before I provide any recommendations. It's important to be interactive and engaging.\n"
    "Here are some resources I might reference during our conversation:\n"
    "1. [iFixit](https://www.ifixit.com) – Comprehensive repair guides for various devices.\n"
    "2. [Tom's Hardware](https://www.tomshardware.com) – Tech news and reviews to help troubleshoot issues.\n"
    "3. [CNET](https://www.cnet.com) – Product reviews and recommendations for tech gadgets.\n"
    "4. [TechSpot](https://www.techspot.com) – News and guides on PC hardware and software troubleshooting.\n"
    "5. [Android Authority](https://www.androidauthority.com) – Advice and tips on smartphone issues and repairs.\n\n"
    "Disclaimer: While I strive to provide helpful and accurate information based on your technical issues and the latest tech resources, "
    "I am an AI model and not a certified technician. My responses are intended to offer general advice and support. "
    "For in-depth diagnosis and personalized technical assistance, please consult a qualified technician or technical support service.\n\n"
    "Let's start by collecting your details. Afterward, I will investigate your issues and offer guidance or next steps based on the information gathered from these reputable sources.\n\n"
)

)

class DiagnosisChatbotView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        Handle user interactions with the doctor AI chatbot for PC diagnosis using Gemini AI.
        Persist chat history and associate each chat with a user.
        """
        print(request.data)
        user_message = request.data.get('message', '')
        chat_id = request.data.get('chat_id')
        image = request.FILES.get('image')  # Retrieve the uploaded image if present
        user = request.user  # Get the authenticated user

        # Validate input
        if not user_message and not image:
            return Response({'error': 'No message or image provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Retrieve or create a new chat for this user
            chat = self.get_or_create_chat(user, chat_id)

            # Initialize response message
            chatbot_response = ""

            # Process image input
            if image:
                with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                    for chunk in image.chunks():
                        temp_file.write(chunk)
                    temp_file.flush()  # Ensure all data is written to the file
                # Upload the image and get the URL
                image_url = upload_image_to_backblaze(image)  # Implement this function in utils.py
                if not image_url:
                    return Response({'error': 'Failed to upload image'}, status=status.HTTP_400_BAD_REQUEST)

                # Create a diagnosis entry if an image is provided
                diagnosis_result = self.get_diagnosis_from_image(image_url)  # Use the image URL for diagnosis
                Diagnosis.objects.create(user=user, symptoms=user_message, diagnosis_result=diagnosis_result, image_url=image_url)

                # Prepare chatbot response with diagnosis
                chatbot_response = f"Diagnosis based on the image: {diagnosis_result}"

                # Save the user message with image URL
                ChatMessage.objects.create(chat=chat, sender='user', message=user_message, image_url=image_url)

            # Process text input only
            if user_message:
                # Save the user message to the chat
                ChatMessage.objects.create(chat=chat, sender='user', message=user_message)

                # Retrieve chat history
                chat_history = self.get_chat_history(chat)

                # Get the chatbot response from Gemini AI
                chatbot_response = self.get_chatbot_response(user_message, chat_history)

            # Save the chatbot response to the chat
            ChatMessage.objects.create(chat=chat, sender='bot', message=chatbot_response)

            # Return the response along with the chat ID for future messages
            return Response({
                'response': chatbot_response,
                'chat_id': chat.id  # Send the chat ID back to the frontend
            }, status=status.HTTP_200_OK)

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            logger.error(f"Error in chatbot interaction: {e}")
            return Response({'error': 'Something went wrong while processing your request.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get_or_create_chat(self, user, chat_id):
        """
        Retrieve an existing chat or create a new one if no chat ID is provided.
        """
        if chat_id:
            try:
                chat = Chat.objects.get(id=chat_id, user=user)
            except Chat.DoesNotExist:
                raise ValidationError('Chat not found or you do not have permission to access it.')
        else:
            chat = Chat.objects.create(user=user)  # Create a new chat for the user
        return chat

    def get_chat_history(self, chat, limit=20):
        """
        Retrieve and format the chat history, limit to last N messages for efficiency.
        """
        messages = ChatMessage.objects.filter(chat=chat).order_by('-timestamp')[:limit]
        messages = reversed(messages)  # Reverse to keep the order as oldest-to-newest
        history = []
        for message in messages:
           role = 'user' if message.sender == 'user' else 'model'
           history.append({
                "role": role,
                "parts": [message.message],
            })
        return history

    def get_chatbot_response(self, user_message, history):
        """
        Send the user's message and chat history to Gemini AI to get a response.
        """
        try:
            chat = model.start_chat(history=history)  # Replace with Gemini AI interaction
            response = chat.send_message(user_message)
            return response.text  # Return the chatbot's textual response
        except Exception as e:
            logger.error(f"Error getting chatbot response from Gemini AI: {e}")
            return "Sorry, I couldn't process your request at this time."

    def get_diagnosis_from_image(self, image_url):
     """
     Analyze the provided image URL for visible hardware and software issues.
    """
     chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [f"Analyze the image at {image_url} for any hardware or software issues."],
            },
            {
                "role": "model",
                "parts": ["Analyzing the image for hardware and software issues..."],
            },
        ]
    )

    # Sending a prompt to the model to get a response
     response = chat_session.send_message("Please provide a detailed diagnosis of any issues.")
     diagnosis_result = response.text  # Capture the diagnosis result
     return diagnosis_result



class UserChatsView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure only logged-in users can access

    def get(self, request):
        user = request.user  # Get the logged-in user
        chats = Chat.objects.filter(user=user)  # Fetch chats associated with the user
        serializer = ChatSerializer(chats, many=True)  # Serialize the chat data
        return Response(serializer.data)  # Return the serialized data
    


# views.py




class PasswordResetRequestView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get('email')
        try:
            # Find user by email
            user = User.objects.get(email=email)
            # Generate token
            token = default_token_generator.make_token(user)
            # Create reset link
            reset_link = f"https://pcairepair.vercel.app/auth/password-reset-confirm?token={token}&email={user.email}"

            # Prepare email content
            email_subject = "Password Reset Request"
            email_body = (
                f"Hello {user.username},\n\n"
                "You requested a password reset. Click the link below to reset your password:\n\n"
                f"{reset_link}\n\n"
                "If you didn't request this, please ignore this email.\n\n"
                "Best regards,\nThe DocTech Team"
            )

            # Send the email
            send_mail(
                subject=email_subject,
                message=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
            )

            return Response({'message': 'Password reset link has been sent to your email.'}, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            return Response({'error': 'User with this email does not exist.'}, status=status.HTTP_400_BAD_REQUEST)
class SignupView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        username = request.data.get('username')
        email = request.data.get('email')
        password = request.data.get('password')

        # Check for required fields
        if not username or not email or not password:
            return Response({'error': 'Username, email, and password are required'}, status=status.HTTP_400_BAD_REQUEST)

        # Check if username or email is already taken
        if User.objects.filter(username=username).exists():
            return Response({'error': 'Username is already taken'}, status=status.HTTP_400_BAD_REQUEST)
        if User.objects.filter(email=email).exists():
            return Response({'error': 'Email is already registered'}, status=status.HTTP_400_BAD_REQUEST)

        # Create the user (inactive until email is verified)
        user = User(username=username, email=email, is_active=False)
        user.set_password(password)
        user.save()

        # Send verification email
        try:
            token = default_token_generator.make_token(user)
            verification_url = f'https://pcairepair.vercel.app/verify-email/?email={user.email}&token={token}'
            print(verification_url)

            email_subject = 'Verify your email address'
            email_body = f'Hello {user.username},\n\n' \
                         f'Thank you for registering with us! Please verify your email address by clicking the link below:\n\n' \
                         f'{verification_url}\n\n' \
                         f'If you didn’t request this, please ignore this email.\n\n' \
                         f'Best regards,\nThe DocTech Team'

            # Prepare email message
            send_mail(
                subject=email_subject,
                message=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
            )

            return Response({'message': 'User registered successfully! Please verify your email.'}, status=status.HTTP_201_CREATED)

        except Exception as e:
            user.delete()  # Ensure user is not saved if email sending fails
            return Response({'error': f'Failed to send verification email: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class VerifyEmailView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get('email')
        token = request.data.get('token')
        
        print(f"Verifying user with email: {email} and token: {token}")  # Debugging line

        try:
            user = User.objects.get(email=email)
            print(f"User found: {user.username}")  # Debugging line
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            return Response('User not found', status=status.HTTP_404_NOT_FOUND)

        if not default_token_generator.check_token(user, token):
            print("Token is invalid or expired.")  # Debugging line
            return Response('Token is invalid or expired. Please request another confirmation email by signing in.', status=status.HTTP_400_BAD_REQUEST)

        user.is_active = True
        user.save()

        # Send welcome email
        try:
            login_url = f'https://pcairepair.vercel.app/auth/login'
            email_subject = 'Welcome to DocTech!'
            email_body = f'Hello {user.username},\n\n' \
                         f'Welcome to DocTech! We’re excited to have you on board. Click the link below to log in:\n\n' \
                         f'{login_url}\n\n' \
                         f'Best regards,\nThe DocTech Team'

            send_mail(
                subject=email_subject,
                message=email_body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
            )

            return Response('Email successfully confirmed', status=status.HTTP_200_OK)

        except Exception as e:
            return Response(f'Failed to send welcome email: {str(e)}', status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PasswordResetConfirmView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        token = request.data.get('token')
        email = request.data.get('email')
        new_password = request.data.get('new_password')
        
        try:
            user = User.objects.get(email=email)
            if default_token_generator.check_token(user, token):
                user.set_password(new_password)
                user.save()
                return Response({'message': 'Password has been reset successfully.'}, status=status.HTTP_200_OK)
            else:
                return Response({'error': 'Invalid token.'}, status=status.HTTP_400_BAD_REQUEST)
        except User.DoesNotExist:
            return Response({'error': 'Invalid user.'}, status=status.HTTP_400_BAD_REQUEST)
        

class ContactUsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        full_name = request.data.get('full_name')
        email = request.data.get('email')
        message = request.data.get('message')

        # Validate the input
        if not full_name or not email or not message:
            return Response({'error': 'Full name, email, and message are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Send the email to the admin/support team
            admin_subject = f"Contact Us Form Submission from {full_name}"
            admin_message = render_to_string('admin_contact_email.html', {
                'full_name': full_name,
                'email': email,
                'message': message,
            })

            admin_email = EmailMultiAlternatives(
                subject=admin_subject,
                body=f"Full Name: {full_name}\nEmail: {email}\nMessage:\n{message}",  # Fallback plain text
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[settings.DEFAULT_FROM_EMAIL],  # The support/admin email
            )
            admin_email.attach_alternative(admin_message, "text/html")
            admin_email.send(fail_silently=False)

        except Exception as e:
            return Response({'error': f'Failed to send message to admin: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            # Send an acknowledgment email to the user
            user_subject = "We Received Your Contact Request"
            user_message = render_to_string('user_contact_acknowledgement.html', {
                'full_name': full_name,
            })

            user_email = EmailMultiAlternatives(
                subject=user_subject,
                body="Thank you for contacting us. We've received your message and will get back to you soon.",  # Fallback plain text
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[email],  # The user's email
            )
            user_email.attach_alternative(user_message, "text/html")
            user_email.send(fail_silently=False)

            return Response({'message': 'Your message has been sent successfully!'}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': f'Failed to send acknowledgment email to user: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class TechNewsAPIView(APIView):
    permission_classes = [IsAuthenticated]  # Uncomment if you want authentication
    
    def get(self, request):
        # Prepare the request parameters
        api_token = settings.NEWS_API_KEY # Replace with your token if stored in settings
        params = urllib.parse.urlencode({
            'api_token': api_token,
            'categories': 'tech',
            'language':'en',
            'limit': 50,
        })
        
        try:
            # Establish the connection to the News API
            conn = http.client.HTTPSConnection('api.thenewsapi.com')
            conn.request('GET', f'/v1/news/all?{params}')

            # Get the response from the API
            res = conn.getresponse()
            data = res.read()
            
            # Decode and parse the response
            decoded_data = data.decode('utf-8')
            news_data = json.loads(decoded_data)  # Convert the string to a dictionary

            # Extract articles or handle empty results
            articles = news_data.get('data', [])
            print(articles)
            
            if not articles:
                return Response({'message': 'No articles found.'}, status=status.HTTP_204_NO_CONTENT)
            
            # Return the articles in the response
            return Response(articles, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Handle any errors and return an appropriate response
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

# class TechNewsAPIView(APIView):
#     # permission_classes = [IsAuthenticated]
#     def get(self, request):
#         api_key=settings.NEWS_API_KEY
#         # Initialize NewsApiClient with your API key
#         newsapi = NewsApiClient(api_key=api_key)

#         # Fetch top headlines in the technology category
#         try:
#             top_headlines = newsapi.get_top_headlines(
#                 category='technology',
#                 language='en',
#                 country='us'
#             )
#             # print(top_headlines)
#             articles = top_headlines.get('articles', [])
#             # print(articles)
            

#             if not articles:
#                 return Response({'message': 'No valid articles found.'}, status=status.HTTP_204_NO_CONTENT)

#             return Response(articles, status=status.HTTP_200_OK)

#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    # def filter_removed_articles(self, articles):
    #     # Filter articles that contain "[Removed]" in title, description, or content
    #     return [
    #         article for article in articles
    #         if not any("[Removed]" in (article.get('title', ''), 
    #                                    article.get('description', ''), 
    #                                    article.get('content', '')))
    #     ]



class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        logout(request)  # This will clear the session
        return Response({'message': 'Successfully logged out.'}, status=status.HTTP_200_OK)
