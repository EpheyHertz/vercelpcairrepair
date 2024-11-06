from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
from .serializers import SignupSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import MyTokenObtainPairSerializer,ChatSerializer,ChatMessageSerializer,QuestionSerializer, AnswerSerializer, LikeDislikeSerializer, CommentSerializer
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

from django.utils import timezone
import pytz    
from .serializers import UserProfileSerializer
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.db import IntegrityError
from newsapi import NewsApiClient
import io
import requests
import os
import PIL.Image
import tempfile
from django.contrib.auth import logout
import smtplib
from django.db.models import Q
from .models import Chat, ChatMessage, Diagnosis,Question, Answer, LikeDislike, Comment
from rest_framework.exceptions import ValidationError
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from datetime import datetime
from .models import NewsArticle,NewsSource
# from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from django.contrib.auth.tokens import default_token_generator


import google.generativeai as genai
GEMINI_AI_API_KEY=settings.GEMINI_AI_API_KEY
genai.configure(api_key=GEMINI_AI_API_KEY)
from .utils import upload_image_to_backblaze, fetch_news_from_scraper
import logging

logger = logging.getLogger(__name__)
class Welcome(APIView):
    permission_classes = [AllowAny]
    def get(self,request):
        return Response({"message":"Welcome to pc diagnosis apis"})
    



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
    "2. [pc world news to get the latest windows update and news](https://www.pcworld.com/windows/news) – Tech news and reviews to help troubleshoot issues.\n"
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
                # img = PIL.Image.open(image)
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
     response = chat_session.send_message(["Analyze the image provided below for any visible hardware or software issues. Please focus on identifying common problems such as: - Signs of physical damage (e.g., cracks, dents, loose components) - Software errors or warnings displayed in the image - Any unusual configurations or settings visible - General condition of the hardware (e.g., cleanliness, organization). Provide a structured list of findings, including a summary of the overall condition of the equipment.",image_url])
     diagnosis_result = response.text  # Capture the diagnosis result
     return diagnosis_result



class UserChatsView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure only logged-in users can access

    def get(self, request):
        user = request.user  # Get the logged-in user
        chats = Chat.objects.filter(user=user).order_by('-created_at')  # Fetch chats in descending order
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
    # permission_classes=[IsAuthenticated]
    def get(self, request):
        # Get pagination params from request
        page = int(request.GET.get('page', 1))  # Default to page 1
        limit = int(request.GET.get('limit', 50))  # Default to 50 articles per page
        offset = (page - 1) * limit

        # Prepare API request parameters
        api_token = settings.NEWS_API2_KEY
        params = urllib.parse.urlencode({
            'api_token': api_token,
            'categories': 'tech,business',
            'language': 'en',
            'limit': limit,
            'page': page,
        })

        try:
            # Fetch articles from the external API
            conn = http.client.HTTPSConnection('api.thenewsapi.com')
            conn.request('GET', f'/v1/news/all?{params}')
            res = conn.getresponse()
            data = res.read()

            decoded_data = data.decode('utf-8')
            news_data = json.loads(decoded_data)
            api_articles = news_data.get('data', [])
            scrapped_articles=fetch_news_from_scraper()
            all_articles_from_apis=api_articles+scrapped_articles
           
            
            # Fetch articles from the database
            db_articles = list(
                    NewsArticle.objects.all()
                    .order_by('-published_at')  
                    .values(
                        'source__name',
                        'author',
                        'title',
                        'description',
                        'url',
                        'urlToImage',
                        'published_at',
                        'content'
                    ) 
                )

            combined_articles = all_articles_from_apis + db_articles
            # Combine API articles and DB articles
            

            # Pagination handling
            total_articles = len(combined_articles) # Total count from API and DB
            total_pages = (total_articles // limit) + (1 if total_articles % limit else 0)
            has_next = page < total_pages

            return Response({
                'page': page,
                'total_pages': total_pages,
                'has_next': has_next,
                'articles': combined_articles,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class TechNewsAPIViewLocalHost(APIView):
    # permission_classes = [IsAuthenticated]

    def get(self, request):
        api_key = settings.NEWS_API_KEY
        newsapi = NewsApiClient(api_key=api_key)

        # Fetch top headlines in the technology category
        try:
            tech_headlines = newsapi.get_everything(
               q='technology',
                language='en',
                sort_by='publishedAt'
            )
            coding_news = newsapi.get_everything(
                q='programming',
                language='en',
                sort_by='publishedAt'  # You can also use 'publishedAt' to get the latest news
            )
            
            # coding_headlines = newsapi.get_everything(
            #     category='technology',
            #     language='en',
            #     country='us'
            # )
            
            # business_headlines =newsapi.get_everything(
            #     q='business',
            #     language='en',
            #     sort_by='relevancy'
            # )

            # Extract articles from both responses
            tech_articles = tech_headlines.get('articles', [])[:50]
            coding_articles = coding_news.get('articles', [])[:50]

            #business_articles = business_headlines.get('articles', [])

            # Combine the articles
            articles= tech_articles + coding_articles 
            # articles = top_headlines.get('articles', [])
            print(articles)
            
            if not articles:
                return Response({'message': 'No valid articles found.'}, status=status.HTTP_204_NO_CONTENT)

            # Filter articles if necessary
            # articles = self.filter_removed_articles(articles)

            # Save articles to the database
            self.save_articles(articles)

            return Response(articles, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def filter_removed_articles(self, articles):
        # Filter articles that contain "[Removed]" in title, description, or content
        return [
            article for article in articles
            if not any("[Removed]" in (article.get('title', ''), 
                                       article.get('description', ''), 
                                       article.get('content', '')))
        ]

    def save_articles(self, articles):
        for article in articles:
            # Extract and format the fields from the API response
            source_data = article.get('source', {})
            source_name = source_data.get('name', None)
            source_id = source_data.get('id', None)

            author = article.get('author', None)
            title = article.get('title', None)
            description = article.get('description', None)
            url = article.get('url', None)
            url_to_image = article.get('urlToImage', None)
            published_at = article.get('publishedAt', None)
            content = article.get('content', None)

            # Skip if the title is longer than 255 characters
            if title and len(title) > 255:
                print(f"Skipping article with title: {title} - Title exceeds 255 characters.")
                continue
            if author and len(author) > 255 :
                print(f"Skipping article with title: {author} - Author name exceeds 255 characters.")
                continue

            # Convert published_at to a timezone-aware datetime object if it's available
            if published_at and isinstance(published_at, str):
                naive_dt = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
                aware_dt = timezone.make_aware(naive_dt, timezone=pytz.UTC)
                published_at = aware_dt

            # Handle the source - either create or get the existing source
            source, created = NewsSource.objects.get_or_create(
                name=source_name,
                defaults={'source_id': source_id}
            )

            # Save each article to the database, preventing duplicates using 'title'
            NewsArticle.objects.get_or_create(
                title=title,
                defaults={
                    'source': source,
                    'author': author,
                    'description': description,
                    'url': url,
                    'urlToImage': url_to_image,  # Consistent field naming
                    'published_at': published_at,
                    'content': content,
                }
            )



class UserProfileAPIView(APIView):
    def get(self, request, pk):
        try:
            user = User.objects.get(pk=pk)  # Get user by primary key
        except User.DoesNotExist:
            return Response({"detail": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        # Serialize user profile
        profile_serializer =UserProfileSerializer(user.profile)
        
        # Get user's questions and answers
        questions = Question.objects.filter(user=user)
        answers = Answer.objects.filter(user=user)

        # Serialize questions and answers
        question_serializer = QuestionSerializer(questions, many=True)
        answer_serializer = AnswerSerializer(answers, many=True)

        return Response({
            'profile': profile_serializer.data,
            'questions': question_serializer.data,
            'answers': answer_serializer.data
        }, status=status.HTTP_200_OK)


class QuestionAPIView(APIView):
    permission_classes=[AllowAny]
    def get(self, request):
        questions = Question.objects.all().order_by('-created_at')
        serializer = QuestionSerializer(questions, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = QuestionSerializer(data=request.data)
        if serializer.is_valid():
            image = request.FILES.get('image')
            image_url = None
            if image:
                image_url = upload_image_to_backblaze(image)
            serializer.save(user=request.user, image_url=image_url)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk):
        try:
            question = Question.objects.get(pk=pk)
        except Question.DoesNotExist:
            return Response({"error": "Question not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = QuestionSerializer(question, data=request.data, partial=True)
        if serializer.is_valid():
            image = request.FILES.get('image')
            image_url = question.image_url
            if image:
                image_url = upload_image_to_backblaze(image)
            serializer.save(image_url=image_url)
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

class AnswerAPIView(APIView):
    def get(self, request):
        answers = Answer.objects.all().order_by('-created_at')
        serializer = AnswerSerializer(answers, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = AnswerSerializer(data=request.data)
        if serializer.is_valid():
            image = request.FILES.get('image')
            image_url = None
            if image:
                image_url = upload_image_to_backblaze(image)
            serializer.save(user=request.user, image_url=image_url)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def put(self, request, pk):
        try:
            answer = Answer.objects.get(pk=pk)
        except Answer.DoesNotExist:
            return Response({"error": "Answer not found"}, status=status.HTTP_404_NOT_FOUND)

        serializer = AnswerSerializer(answer, data=request.data, partial=True)
        if serializer.is_valid():
            image = request.FILES.get('image')
            image_url = answer.image_url
            if image:
                image_url = upload_image_to_backblaze(image)
            serializer.save(image_url=image_url)
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LikeDislikeAPIView(APIView):
    def post(self, request):
        serializer = LikeDislikeSerializer(data=request.data)
        if serializer.is_valid():
            user = request.user
            question = serializer.validated_data.get('question', None)
            answer = serializer.validated_data.get('answer', None)

            if question:
                # Check if question exists
                try:
                    question_instance = Question.objects.get(id=question.id)
                except Question.DoesNotExist:
                    return Response({"error": "Question not found."}, status=status.HTTP_404_NOT_FOUND)

                # Check for existing vote
                existing_vote = LikeDislike.objects.filter(user=user, question=question_instance).first()
                if existing_vote:
                    if existing_vote.vote == serializer.validated_data['vote']:
                        return Response({"error": "You have already voted on this question."}, status=status.HTTP_400_BAD_REQUEST)
                    else:
                        # Update existing vote
                        existing_vote.vote = serializer.validated_data['vote']
                        existing_vote.save()
                        return Response({"message": "Vote on question updated."}, status=status.HTTP_200_OK)
                else:
                    # Save new vote
                    serializer.save(user=user, question=question_instance)
                    return Response({"message": "Vote on question saved."}, status=status.HTTP_201_CREATED)

            elif answer:
                # Check if answer exists
                try:
                    answer_instance = Answer.objects.get(id=answer.id)
                except Answer.DoesNotExist:
                    return Response({"error": "Answer not found."}, status=status.HTTP_404_NOT_FOUND)

                # Check for existing vote
                existing_vote = LikeDislike.objects.filter(user=user, answer=answer_instance).first()
                if existing_vote:
                    if existing_vote.vote == serializer.validated_data['vote']:
                        return Response({"error": "You have already voted on this answer."}, status=status.HTTP_400_BAD_REQUEST)
                    else:
                        # Update existing vote
                        existing_vote.vote = serializer.validated_data['vote']
                        existing_vote.save()
                        return Response({"message": "Vote on answer updated."}, status=status.HTTP_200_OK)
                else:
                    # Save new vote
                    serializer.save(user=user, answer=answer_instance)
                    return Response({"message": "Vote on answer saved."}, status=status.HTTP_201_CREATED)

            return Response({"error": "You must vote on either a question or an answer."}, status=status.HTTP_400_BAD_REQUEST)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class CommentAPIView(APIView):
    def get(self, request):
        comments = Comment.objects.all().order_by('-created_at')
        serializer = CommentSerializer(comments, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = CommentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SearchAPIView(APIView):
    def get(self, request):
        query = request.query_params.get('q',None)
        if not query:
            return Response({"error": "No search query provided."}, status=status.HTTP_400_BAD_REQUEST)

        questions = Question.objects.filter(
            Q(title__icontains=query) | Q(content__icontains=query) | Q(user__username__icontains=query)
        )
        answers = Answer.objects.filter(
            Q(content__icontains=query) | Q(user__username__icontains=query)
        )

        question_serializer = QuestionSerializer(questions, many=True)
        answer_serializer = AnswerSerializer(answers, many=True)

        return Response({
            'questions': question_serializer.data,
            'answers': answer_serializer.data
        }, status=status.HTTP_200_OK)


class FollowUserAPIView(APIView):
    def post(self, request, pk):
        user_to_follow = User.objects.get(pk=pk)
        if user_to_follow == request.user:
            return Response({"error": "You cannot follow yourself."}, status=status.HTTP_400_BAD_REQUEST)

        if request.user.profile.following.filter(pk=pk).exists():
            return Response({"error": "You are already following this user."}, status=status.HTTP_400_BAD_REQUEST)

        request.user.profile.following.add(user_to_follow)
        return Response({"message": f"You are now following {user_to_follow.username}"}, status=status.HTTP_200_OK)

    def delete(self, request, pk):
        user_to_unfollow = User.objects.get(pk=pk)
        if user_to_unfollow == request.user:
            return Response({"error": "You cannot unfollow yourself."}, status=status.HTTP_400_BAD_REQUEST)

        if not request.user.profile.following.filter(pk=pk).exists():
            return Response({"error": "You are not following this user."}, status=status.HTTP_400_BAD_REQUEST)

        request.user.profile.following.remove(user_to_unfollow)
        return Response({"message": f"You have unfollowed {user_to_unfollow.username}"}, status=status.HTTP_200_OK)

class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        logout(request)  # This will clear the session
        return Response({'message': 'Successfully logged out.'}, status=status.HTTP_200_OK)
