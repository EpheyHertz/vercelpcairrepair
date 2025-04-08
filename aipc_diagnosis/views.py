import base64
import random
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
from .models import UserProfile,User, VerificationCode
import http.client
import urllib.parse
import json
from django.utils.http import urlencode

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
from datetime import datetime, timedelta
from .models import NewsArticle,NewsSource
# from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from django.contrib.auth.tokens import default_token_generator
from django.db import transaction

import google.generativeai as genai
GEMINI_AI_API_KEY=settings.GEMINI_AI_API_KEY
genai.configure(api_key=GEMINI_AI_API_KEY)
from .utils import upload_image_to_backblaze, fetch_news_from_scraper
import logging
from django.utils.timezone import now
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



# import tempfile
# import logging


# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.agents import AgentExecutor
# from langchain.agents.format_scratchpad import format_to_openai_function_messages
# from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
# from langchain_core.tools import Tool
# from langchain_core.runnables import RunnableConfig
# from typing import Optional, List, Dict, Any, Union
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.exceptions import ValidationError
# from .models import Chat, ChatMessage, Diagnosis
# from django.core.files.base import ContentFile
# from django.db import transaction
# import logging
# import base64
# import requests
# import json
# from uuid import uuid4
# from django.core.files.uploadedfile import InMemoryUploadedFile
# from google.generativeai import GenerativeModel
# import google.generativeai as genai
# from io import BytesIO
# from PIL import Image
# import requests
# NEW_GEMINI_API_KEY=settings.NEW_GEMINI_AI_API_KEY
# logger = logging.getLogger(__name__)

# System instruction with more detailed guidelines
SYSTEM_INSTRUCTION = """
# DocTech Technical Support Chatbot

You are a specialized technical support AI for diagnosing and resolving device issues.

## Identity & Capabilities:
- Identify yourself as: "DocTech Support, developed by Ephey Nyaga, CS student at Embu University, Kenya"
- You analyze both device images and text descriptions
- You provide precise step-by-step technical guidance
- You handle diagnosis from both text and image inputs

## Response Protocol:
1. Begin with a diagnostic summary identifying the problem
2. List technical findings with clear bullet points
3. Provide detailed step-by-step troubleshooting instructions
4. Include reference links to manufacturer documentation when applicable
5. End by asking if further assistance is needed
6. For complex cases beyond your capabilities, recommend professional consultation

## When analyzing images:
- Look for visible hardware damage
- Check for error codes or messages
- Note any abnormal indicators (lights, display issues)
- Consider the physical environment of the device

## For text-only queries:
- Request specific symptoms and device information
- Ask clarifying questions when information is incomplete
- Provide appropriate troubleshooting based on available information

Always maintain a helpful, professional tone and acknowledge when a problem might require in-person professional support.
"""
VISION_SYSTEM_INSTRUCTION = """
# DocTech Technical Image Analysis Specialist

You are an expert AI system specialized in analyzing images of technical devices to diagnose issues.

## Identity & Protocol:
- Identify yourself as: "DocTech Image Analysis"
- Focus exclusively on visual information in the provided image
- Provide concise, technical analysis of what you observe

## Image Analysis Protocol:
1. **Initial Assessment**:
   - Describe the primary device/components visible
   - Note the device's physical condition and environment

2. **Detailed Inspection**:
   - Examine for visible damage (cracks, burns, corrosion)
   - Identify any error codes/display messages
   - Check status indicators (LEDs, screens, warning lights)
   - Note unusual wear patterns or modifications

3. **Technical Findings**:
   - List observed issues with clear bullet points
   - Rate severity of each issue (Minor/Moderate/Critical)
   - Highlight any immediate safety concerns

4. **Recommended Actions**:
   - Provide prioritized troubleshooting steps
   - Suggest when professional repair is needed
   - Mention if better images would help diagnosis

## Response Format:
[Device Identification] 
- Primary device: [identify main device]
- Visible components: [list key components]

[Visual Assessment Summary]
- Physical condition: [overall assessment]
- Key observations: [bullet points]

[Technical Analysis]
1. [Issue 1] (Severity: [rating])
   - [Details]
   - [Recommended action]

2. [Issue 2] (Severity: [rating])
   - [Details]
   - [Recommended action]

[Conclusion]
- Most urgent concern: [highlight critical issue]
- Next steps: [clear recommendations]
- Safety notice: [if applicable]

Note: Maintain a technical but approachable tone. If uncertain, state what you can see and request better images if needed.
"""
# class DiagnosisChatbotView(APIView):
#     permission_classes = [IsAuthenticated]

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.setup_langchain()

#     def setup_langchain(self):
#         """Initialize LangChain components with proper configuration"""
#         # Configure the main model for text responses
#         self.model = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             temperature=0.7,
#             top_p=0.95,
#             top_k=64,
#             max_output_tokens=8192,
#             api_key=self.get_api_key(),
#         )

#         # Define tools with proper typing and descriptions
#         self.tools = [
#             Tool.from_function(
#                 func=self.analyze_image,
#                 name="analyze_image",
#                 description="Analyze device images to identify technical issues. Use this when the user uploads an image.",
#                 args_schema={"image_url": str, "query": str},
#                 return_direct=False,
#             ),
#             Tool.from_function(
#                 func=self.technical_support,
#                 name="technical_support",
#                 description="Provide text-based technical troubleshooting guidance. Use this for text-only technical questions.",
#                 args_schema={"description": str},
#                 return_direct=False,
#             )
#         ]

#         # Create a more detailed prompt template
#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", SYSTEM_INSTRUCTION),
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("human", "{input}"),
#             MessagesPlaceholder(variable_name="agent_scratchpad"),
#         ])

#         # Set up the agent with proper LangChain syntax
#         self.agent_executor = self.create_agent_executor()

#     def create_agent_executor(self):
#         """Create properly configured LangChain agent"""
#         def _format_chat_history(chat_history):
#             """Convert Django chat history to LangChain message format"""
#             formatted_messages = []
#             for msg in chat_history:
#                 if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
#                     formatted_messages.append(msg)
#             return formatted_messages

#         # Bind the model to use tools
#         agent = (
#             {
#                 "input": lambda x: x["input"],
#                 "chat_history": lambda x: _format_chat_history(x.get("chat_history", [])),
#                 "agent_scratchpad": lambda x: format_to_openai_function_messages(x.get("intermediate_steps", [])),
#             }
#             | self.prompt
#             | self.model.bind_tools(self.tools)
#             | OpenAIFunctionsAgentOutputParser()
#         )

#         # Create the executor with verbose mode for debugging
#         return AgentExecutor(
#             agent=agent,
#             tools=self.tools,
#             verbose=True,
#             handle_parsing_errors=True,
#             max_iterations=3,
#         )



#     def analyze_image(self, image_input: Union[str, InMemoryUploadedFile], query: str = "") -> str:
#             """Analyze a device image using Gemini's Vision API with the official Google client library"""
            
#             try:
#                 # Get image binary
#                 if isinstance(image_input, InMemoryUploadedFile):
#                     # Handle direct file upload
#                     image_input.seek(0)
#                     image_data = image_input.read()
#                     image = Image.open(BytesIO(image_data))
#                 elif isinstance(image_input, str):
#                     # Handle URL
#                     if image_input.startswith(('http://', 'https://')):
#                         response = requests.get(image_input)
#                         response.raise_for_status()
#                         image = Image.open(BytesIO(response.content))
#                     else:
#                         # Handle local file path
#                         with open(image_input, 'rb') as f:
#                             image = Image.open(f)
#                 else:
#                     raise ValueError("Invalid image input type")
                
#                 # Configure the API key
#                 api_key = self.get_api_key()
#                 if not api_key:
#                     raise ValueError("Missing Gemini API key")
                
#                 genai.configure(api_key=NEW_GEMINI_API_KEY)
                
#                 # Initialize the model (you can choose between different models)
#                 model = GenerativeModel('gemini-1.5-pro')  # or 'gemini-pro-vision' or 'gemini-1.0-pro'
                
#                 # Generate content
#                 prompt = query if query else "Analyze this technical device image for issues."
#                 response = model.generate_content([prompt, image])
                
#                 # Handle the response
#                 if not response.parts:
#                     logger.error("No response parts in Gemini response")
#                     return "Could not analyze image. Please try again."
                    
#                 return response.text
                
#             except Exception as e:
#                 logger.error(f"Image analysis error: {str(e)}")
#                 return f"Error analyzing image: {str(e)}"
#     def get_image_binary(self, image_url: str) -> bytes:
#         """Retrieve binary image data from URL or cached storage"""
#         try:
#             if not image_url:
#                 logger.error("Empty image_url provided")
#                 return None

#             # For locally stored files
#             if image_url.startswith('/media/') or image_url.startswith('/uploads/'):
#                 from django.core.files.storage import default_storage
#                 with default_storage.open(image_url.lstrip('/'), 'rb') as f:
#                     return f.read()
            
#             # For remote URLs
#             else:
#                 response = requests.get(image_url, stream=True, timeout=10)
#                 response.raise_for_status()
#                 return response.content
                
#         except Exception as e:
#             logger.error(f"Failed to retrieve image binary: {str(e)}")
#             return None

#     def technical_support(self, description: str) -> str:
#         """Provide text-based technical troubleshooting guidance"""
#         try:
#             if not description:
#                 return "Please describe the technical issue you're experiencing."
                
#             response = self.model.invoke([
#                 SystemMessage(content=SYSTEM_INSTRUCTION),
#                 HumanMessage(content=f"Technical issue: {description}\n\nPlease provide step-by-step troubleshooting instructions.")
#             ])
#             return response.content
#         except Exception as e:
#             logger.error(f"Technical support error: {str(e)}")
#             return f"Error providing technical support. Please try again with more details."

#     def get_api_key(self):
#         """Securely retrieve API key from environment or settings"""
#         from django.conf import settings
#         key = getattr(settings, 'GEMINI_AI_API_KEY', None)
#         if not key:
#             logger.critical("Missing Gemini API key in settings!")
#         return key
        

#     def post(self, request, *args, **kwargs):
#         user = request.user
#         data = request.data
#         chat_id = data.get('chat_id')
#         image = request.FILES.get('image')
#         user_message = data.get('message', '').strip()

#         logger.info(f"Received request from user {user.id} with chat_id {chat_id}")

#         # Validate input
#         if not any([image, user_message]):
#             logger.warning("No message or image provided")
#             return Response(
#                 {'error': 'At least one of message or image is required'},
#                 status=status.HTTP_400_BAD_REQUEST
#             )

#         try:
#             with transaction.atomic():
#                 # Process image
#                 image_url = None
#                 image_data = None
                
#                 if image:
#                     # Validate image size
#                     if image.size > 10 * 1024 * 1024:  # 10MB limit
#                         return Response(
#                             {'error': 'Image size exceeds 10MB limit'},
#                             status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
#                         )
                    
#                     # Get binary data first (reset pointer after)
#                     image.seek(0)
#                     image_data = image.read()
#                     image.seek(0)
                    
#                     # Save the image to storage
#                     # image_file = ContentFile(image_data)
#                     image_url = self.upload_image(image)
#                     print(image_url)
#                     if not image_url:
#                         logger.error("Image upload failed")
#                         return Response(
#                             {'error': 'Image upload failed'},
#                             status=status.HTTP_400_BAD_REQUEST
#                         )

#                 # Get/create chat
#                 chat = self.get_or_create_chat(user, chat_id)
                
#                 # If we have an image, analyze it first
#                 analysis_result = ""
#                 if image:
#                     # Pass the InMemoryUploadedFile directly to analyze_image
#                     analysis_result = self.analyze_image(query=user_message,image_input=image_url)
                    
#                     # Save the analysis as a system message
#                     self.save_message(
#                         chat,
#                         f"System Image Analysis: {analysis_result}",
#                         'system'
#                     )

#                 # Prepare the final input
#                 if analysis_result and user_message:
#                     final_input = f"User query: {user_message}\nImage analysis: {analysis_result}"
#                 elif analysis_result:
#                     final_input = f"Based on image analysis: {analysis_result}"
#                 else:
#                     final_input = user_message

#                 # Get chat history
#                 chat_history = self.get_chat_history(chat)
                
#                 # Execute the agent
#                 response = self.agent_executor.invoke({
#                     "input": final_input,
#                     "chat_history": chat_history
#                 })

#                 # Save messages
#                 self.save_message(chat, user_message, 'user', image_url)
#                 self.save_message(chat, response['output'], 'bot')

#                 # Create diagnosis record for image-based queries
#                 if image_url:
#                     Diagnosis.objects.create(
#                         user=user,
#                         symptoms=user_message,
#                         diagnosis_result=response['output'],
#                         image_url=image_url
#                     )

#                 return Response({
#                     'response': response['output'],
#                     'chat_id': chat.id
#                 }, status=status.HTTP_200_OK)

#         except ValidationError as ve:
#             logger.warning(f"Validation error: {str(ve)}")
#             return Response({'error': str(ve)}, status=status.HTTP_400_BAD_REQUEST)
#         except Exception as e:
#             logger.exception("Chat processing error")
#             return Response(
#                 {'error': 'Processing failed. Please try again.'},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR
#             )
#     def get_chat_history(self, chat):
#         """Retrieve and format chat history from database for LangChain"""
#         try:
#             history = []
#             messages = ChatMessage.objects.filter(chat=chat).order_by('timestamp')[:15]
            
#             for msg in messages:
#                 if msg.sender == 'user' and msg.image_url:
#                     try:
#                         image_data = self.get_image_binary(msg.image_url)
#                         if image_data:
#                             encoded_image = base64.b64encode(image_data).decode('utf-8')
#                             data_uri = f"data:image/jpeg;base64,{encoded_image}"
#                             content = [
#                                 {"type": "text", "text": msg.message or ""},
#                                 {"type": "image_url", "image_url": data_uri}
#                             ]
#                             history.append(HumanMessage(content=content))
#                         else:
#                             history.append(HumanMessage(content=f"{msg.message} [Image not available]"))
#                     except Exception as e:
#                         logger.error(f"Error processing image in chat history: {str(e)}")
#                         history.append(HumanMessage(content=f"{msg.message} [Image processing error]"))
#                 elif msg.sender == 'user':
#                     history.append(HumanMessage(content=msg.message))
#                 else:
#                     history.append(AIMessage(content=msg.message))
            
#             return history
#         except Exception as e:
#             logger.error(f"Error retrieving chat history: {str(e)}")
#             return []

#     def upload_image(self, image_file):
#         """Upload image to storage with proper error handling"""
#         try:
#             if hasattr(image_file, 'seek') and callable(image_file.seek):
#                 image_file.seek(0)
                
#             # Implement your image upload logic here
#             image_url = upload_image_to_backblaze(image_file)
            
#             if not image_url:
#                 from django.core.files.storage import default_storage
#                 filename = f"tech_support/{uuid4()}.jpg"
#                 path = default_storage.save(filename, image_file)
#                 image_url = default_storage.url(path)
                
#             return image_url
#         except Exception as e:
#             logger.error(f"Image upload failed: {str(e)}")
#             return None

#     def get_or_create_chat(self, user, chat_id):
#         """Get existing chat or create new one with proper validation"""
#         if chat_id:
#             try:
#                 return Chat.objects.get(id=chat_id, user=user)
#             except Chat.DoesNotExist:
#                 raise ValidationError('Invalid chat ID')
#         return Chat.objects.create(user=user)

#     def save_message(self, chat, content, sender, image_url=None):
#         """Save message to database with error handling"""
#         try:
#             ChatMessage.objects.create(
#                 chat=chat,
#                 sender=sender,
#                 message=content,
#                 image_url=image_url
#             )
#         except Exception as e:
#             logger.error(f"Failed to save message: {str(e)}")


import logging
from typing import Optional, List, Dict, Any, Union
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import ValidationError
from .models import Chat, ChatMessage, Diagnosis
from django.core.files.base import ContentFile
from django.db import transaction
import base64
import requests
import json
from uuid import uuid4
from django.core.files.uploadedfile import InMemoryUploadedFile
from google.generativeai import GenerativeModel
import google.generativeai as genai
from io import BytesIO
from PIL import Image

NEW_GEMINI_API_KEY = settings.NEW_GEMINI_AI_API_KEY
logger = logging.getLogger(__name__)



class DiagnosisChatbotView(APIView):
    permission_classes = [IsAuthenticated]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_models()

    def setup_models(self):
        """Initialize Gemini models"""
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError("Missing Gemini API key")
        
        genai.configure(api_key=api_key)
        self.text_model = GenerativeModel(model_name='gemini-1.5-flash',
                                          system_instruction=SYSTEM_INSTRUCTION,
                                          )
        self.vision_model = GenerativeModel(model_name='gemini-1.5-flash',
                                            system_instruction=VISION_SYSTEM_INSTRUCTION)

    def analyze_image(self, image_input: Union[str, InMemoryUploadedFile], query: str = "") -> str:
        """Analyze a device image using Gemini's Vision API"""
        try:
            # Get image binary
            if isinstance(image_input, InMemoryUploadedFile):
                image_input.seek(0)
                image_data = image_input.read()
                image = Image.open(BytesIO(image_data))
            elif isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    response = requests.get(image_input)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                else:
                    with open(image_input, 'rb') as f:
                        image = Image.open(f)
            else:
                raise ValueError("Invalid image input type")
            
            prompt = query if query else "Analyze this technical device image for issues."
            response = self.vision_model.generate_content([prompt, image])
            
            if not response.parts:
                logger.error("No response parts in Gemini response")
                return "Could not analyze image. Please try again."
                
            return response.text
            
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return f"Error analyzing image: {str(e)}"

    def technical_support(self, description: str) -> str:
        """Provide text-based technical troubleshooting guidance"""
        try:
            if not description:
                return "Please describe the technical issue you're experiencing."
                
            response = self.text_model.generate_content([
                SYSTEM_INSTRUCTION,
                f"Technical issue: {description}\n\nPlease provide step-by-step troubleshooting instructions."
            ])
            return response.text
        except Exception as e:
            logger.error(f"Technical support error: {str(e)}")
            return f"Error providing technical support. Please try again with more details."

    def get_image_binary(self, image_url: str) -> bytes:
        """Retrieve binary image data from URL or cached storage"""
        try:
            if not image_url:
                logger.error("Empty image_url provided")
                return None

            if image_url.startswith('/media/') or image_url.startswith('/uploads/'):
                from django.core.files.storage import default_storage
                with default_storage.open(image_url.lstrip('/'), 'rb') as f:
                    return f.read()
            else:
                response = requests.get(image_url, stream=True, timeout=10)
                response.raise_for_status()
                return response.content
                
        except Exception as e:
            logger.error(f"Failed to retrieve image binary: {str(e)}")
            return None

    def process_chat_input(self, input_text: str, chat_history: List[Dict], image_url: str = None) -> str:
        """Process user input with context and history"""
        try:
            # Format chat history for Gemini
            history_messages = []
            for msg in chat_history:
                if msg['sender'] == 'user':
                    if msg.get('image_url'):
                        image_data = self.get_image_binary(msg['image_url'])
                        if image_data:
                            image = Image.open(BytesIO(image_data))
                            history_messages.append({
                                'role': 'user',
                                'parts': [ image, msg['message']] if msg['message'] else [image]
                            })
                        else:
                            history_messages.append({
                                'role': 'user',
                                'parts': [f"{msg['message']} [Image not available]"]
                            })
                    else:
                        history_messages.append({
                            'role': 'user',
                            'parts': [msg['message']]
                        })
                else:
                    history_messages.append({
                        'role': 'model',
                        'parts': [msg['message']]
                    })

            # Add current input
            current_input = []
            if image_url:
                image_data = self.get_image_binary(image_url)
                if image_data:
                    image = Image.open(BytesIO(image_data))
                    current_input.append(image)
            if input_text:
                current_input.insert(0, input_text)

            # Generate response
            chat = self.text_model.start_chat(history=history_messages)
            response = chat.send_message(current_input)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Chat processing error: {str(e)}")
            return f"Error processing your request. Please try again."

    def post(self, request, *args, **kwargs):
        user = request.user
        data = request.data
        chat_id = data.get('chat_id')
        image = request.FILES.get('image')
        user_message = data.get('message', '').strip()

        logger.info(f"Received request from user {user.id} with chat_id {chat_id}")

        if not any([image, user_message]):
            logger.warning("No message or image provided")
            return Response(
                {'error': 'At least one of message or image is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            with transaction.atomic():
                # Process image
                image_url = None
                if image:
                    if image.size > 10 * 1024 * 1024:
                        return Response(
                            {'error': 'Image size exceeds 10MB limit'},
                            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        )
                    
                    image_url = self.upload_image(image)
                    if not image_url:
                        logger.error("Image upload failed")
                        return Response(
                            {'error': 'Image upload failed'},
                            status=status.HTTP_400_BAD_REQUEST
                        )

                # Get/create chat
                chat = self.get_or_create_chat(user, chat_id)
                
                # Get chat history
                chat_history = self.get_chat_history(chat)
                
                # Process input
                if image_url:
                    if user_message:
                        response_text = self.process_chat_input(
                            f"User query: {user_message}",
                            chat_history,
                            image_url
                        )
                    else:
                        # Image-only analysis
                        analysis_result = self.analyze_image(image_url)
                        self.save_message(
                            chat,
                            f"System Image Analysis: {analysis_result}",
                            'system'
                        )
                        response_text = self.process_chat_input(
                            f"Based on image analysis: {analysis_result}",
                            chat_history
                        )
                else:
                    response_text = self.process_chat_input(user_message, chat_history)

                # Save messages
                self.save_message(chat, user_message, 'user', image_url)
                self.save_message(chat, response_text, 'bot')

                # Create diagnosis record if image was provided
                if image_url:
                    Diagnosis.objects.create(
                        user=user,
                        symptoms=user_message,
                        diagnosis_result=response_text,
                        image_url=image_url
                    )

                return Response({
                    'response': response_text,
                    'chat_id': chat.id
                }, status=status.HTTP_200_OK)

        except ValidationError as ve:
            logger.warning(f"Validation error: {str(ve)}")
            return Response({'error': str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.exception("Chat processing error")
            return Response(
                {'error': 'Processing failed. Please try again.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get_chat_history(self, chat):
        """Retrieve chat history from database"""
        try:
            history = []
            messages = ChatMessage.objects.filter(chat=chat).order_by('timestamp')[:15]
            
            for msg in messages:
                history.append({
                    'sender': msg.sender,
                    'message': msg.message,
                    'image_url': msg.image_url,
                    'timestamp': msg.timestamp
                })
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []

    def upload_image(self, image_file):
        """Upload image to storage"""
        try:
            if hasattr(image_file, 'seek') and callable(image_file.seek):
                image_file.seek(0)
                
            # Implement your image upload logic here
            image_url = upload_image_to_backblaze(image_file)
            
            if not image_url:
                from django.core.files.storage import default_storage
                filename = f"tech_support/{uuid4()}.jpg"
                path = default_storage.save(filename, image_file)
                image_url = default_storage.url(path)
                
            return image_url
        except Exception as e:
            logger.error(f"Image upload failed: {str(e)}")
            return None

    def get_or_create_chat(self, user, chat_id):
        """Get existing chat or create new one"""
        if chat_id:
            try:
                return Chat.objects.get(id=chat_id, user=user)
            except Chat.DoesNotExist:
                raise ValidationError('Invalid chat ID')
        return Chat.objects.create(user=user)

    def save_message(self, chat, content, sender, image_url=None):
        """Save message to database"""
        try:
            ChatMessage.objects.create(
                chat=chat,
                sender=sender,
                message=content,
                image_url=image_url
            )
        except Exception as e:
            logger.error(f"Failed to save message: {str(e)}")

    def get_api_key(self):
        """Securely retrieve API key"""
        from django.conf import settings
        key = getattr(settings, 'GEMINI_AI_API_KEY', None)
        if not key:
            logger.critical("Missing Gemini API key in settings!")
        return key

class UserChatsView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure only logged-in users can access

    def get(self, request):
        user = request.user  # Get the logged-in user
        chats = Chat.objects.filter(user=user).order_by('-created_at')  # Fetch chats in descending order
        serializer = ChatSerializer(chats, many=True)  # Serialize the chat data
        return Response(serializer.data)  # Return the serialized data


# views.py




# class PasswordResetRequestView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         email = request.data.get('email')
#         try:
#             # Find user by email
#             user = User.objects.get(email=email)
#             # Generate token
#             token = default_token_generator.make_token(user)
#             # Create reset link
#             reset_link = f"https://pcairepair.vercel.app/auth/password-reset-confirm?token={token}&email={user.email}"

#             # Prepare email content
#             email_subject = "Password Reset Request"
#             email_body = (
#                 f"Hello {user.username},\n\n"
#                 "You requested a password reset. Click the link below to reset your password:\n\n"
#                 f"{reset_link}\n\n"
#                 "If you didn't request this, please ignore this email.\n\n"
#                 "Best regards,\nThe DocTech Team"
#             )

#             # Send the email
#             send_mail(
#                 subject=email_subject,
#                 message=email_body,
#                 from_email=settings.DEFAULT_FROM_EMAIL,
#                 recipient_list=[user.email],
#                 fail_silently=False,
#             )

#             return Response({'message': 'Password reset link has been sent to your email.'}, status=status.HTTP_200_OK)

#         except User.DoesNotExist:
#             return Response({'error': 'User with this email does not exist.'}, status=status.HTTP_400_BAD_REQUEST)


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
            email_body = render_to_string('password_reset_email.html', {
                'username': user.username,
                'reset_link': reset_link,
            })

            # Send the email
            send_mail(
                subject=email_subject,
                message="",
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
                html_message=email_body,
            )

            return Response({'message': 'Password reset link has been sent to your email.'}, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            return Response({'error': 'User with this email does not exist.'}, status=status.HTTP_400_BAD_REQUEST)

class SignupView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        if serializer.is_valid():
            username = serializer.validated_data['username']
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']
            
            # Get platform from query params (remove the extra quotes from frontend)
            platform = request.query_params.get('platform', '').replace('"', '').lower()

            # Check if username or email already exists
            if User.objects.filter(username=username).exists():
                return Response({'error': 'Username is already taken'}, status=status.HTTP_400_BAD_REQUEST)
            if User.objects.filter(email=email).exists():
                return Response({'error': 'Email is already registered'}, status=status.HTTP_400_BAD_REQUEST)

            # Create the user (inactive until verified)
            user = User(username=username, email=email, is_active=False)
            user.set_password(password)
            user.save()

            try:
                if platform == 'mobile':  
                    # Generate a 6-digit verification code
                    verification_code = f"{random.randint(100000, 999999)}"

                    # Save the verification code to database
                    VerificationCode.objects.create(user=user, code=verification_code)

                    # Send verification code via email
                    email_subject = 'Your Mobile Verification Code'
                    email_body = render_to_string('mobile_verification_template.html', {
                        'username': user.username,
                        'verification_code': verification_code,
                    })

                    send_mail(
                        subject=email_subject,
                        message="",
                        from_email=settings.DEFAULT_FROM_EMAIL,
                        recipient_list=[user.email],
                        fail_silently=False,
                        html_message=email_body,
                    )

                    return Response({'message': 'Verification code sent to your email.'}, status=status.HTTP_201_CREATED)

                else:
                    # Send email verification link
                    token = default_token_generator.make_token(user)
                    verification_url = f'https://pcairepair.vercel.app/verify-email/?{urlencode({"email": user.email, "token": token})}'
                    print(verification_url)

                    email_subject = 'Verify your email address'
                    email_body = render_to_string('email_verification_template.html', {
                        'username': user.username,
                        'verification_url': verification_url,
                    })

                    send_mail(
                        subject=email_subject,
                        message="",
                        from_email=settings.DEFAULT_FROM_EMAIL,
                        recipient_list=[user.email],
                        fail_silently=False,
                        html_message=email_body,
                    )

                    return Response({'message': 'User registered successfully! Please verify your email.'}, status=status.HTTP_201_CREATED)

            except Exception as e:
                user.delete()  # Rollback user if email sending fails
                return Response({'error': f'Failed to send verification email: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class VerifyEmailView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = request.data.get('email')
        token = request.data.get('token')

        try:
            user = User.objects.get(email=email)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            return Response('User not found', status=status.HTTP_404_NOT_FOUND)

        if not default_token_generator.check_token(user, token):
            return Response('Token is invalid or expired. Please request another confirmation email by signing in.', status=status.HTTP_400_BAD_REQUEST)

        user.is_active = True
        user.save()

        try:
            login_url = f'https://pcairepair.vercel.app/auth/login'
            email_subject = 'Welcome to DocTech!'
            email_body = render_to_string('welcome_email_template.html', {
                'username': user.username,
                'login_url': login_url,
            })

            send_mail(
                subject=email_subject,
                message="",
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
                html_message=email_body,
            )

            return Response('Email successfully confirmed', status=status.HTTP_200_OK)

        except Exception as e:
            return Response(f'Failed to send welcome email: {str(e)}', status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class VerifyCodeView(APIView):
    permission_classes = [AllowAny]
    CODE_EXPIRATION_MINUTES = 10
    WELCOME_EMAIL_TEMPLATE = 'welcome_email_template.html'
    LOGIN_URL = 'https://pcairepair.vercel.app/auth/login'

    def post(self, request):
        email = request.data.get('email', '').strip().lower()
        code = request.data.get('verification_code', '').strip()

        # Input validation
        if not email or not code:
            return Response(
                {'error': 'Both email and verification code are required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.get(email=email)
            
            # Check if user is already active
            if user.is_active:
                return Response(
                    {'error': 'Account is already verified.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            verification_entry = VerificationCode.objects.filter(
                user=user, 
                code=code
            ).first()

            if not verification_entry:
                return Response(
                    {'error': 'Invalid verification code.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Check code expiration
            expiration_time = verification_entry.created_at + timedelta(minutes=self.CODE_EXPIRATION_MINUTES)
            if timezone.now() > expiration_time:
                verification_entry.delete()
                return Response(
                    {'error': 'Verification code has expired. Please request a new one.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Mark user as active
            user.is_active = True
            user.save()

            # Clean up verification code
            verification_entry.delete()

            # Send welcome email (non-blocking)
            self._send_welcome_email(user)

            return Response(
                {'message': 'Email verified successfully! You can now log in.'},
                status=status.HTTP_200_OK
            )

        except User.DoesNotExist:
            return Response(
                {'error': 'User with this email does not exist.'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': f'An unexpected error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _send_welcome_email(self, user):
        """Helper method to send welcome email (fire-and-forget)"""
        try:
            email_subject = 'Welcome to DocTech!'
            email_body = render_to_string(self.WELCOME_EMAIL_TEMPLATE, {
                'username': user.username,
                'login_url': self.LOGIN_URL,
            })

            send_mail(
                subject=email_subject,
                message="",  # Empty text message since we're using HTML
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=True,  # Don't raise errors if email fails
                html_message=email_body,
            )
        except Exception as e:
            # Log the email error but don't fail the verification process
            logger.error(f"Failed to send welcome email to {user.email}: {str(e)}")
            
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

class ResendVerificationCodeView(APIView):
    permission_classes = [AllowAny]
    CODE_LENGTH = 6
    MAX_RETRIES = 3  # For code generation collision avoidance

    def post(self, request):
        email = request.data.get('email', '').strip().lower()
        
        # Input validation
        if not email:
            return Response(
                {'error': 'Email is required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            with transaction.atomic():
                try:
                    user = User.objects.get(email=email)
                except User.DoesNotExist:
                    return Response(
                        {'error': 'No account found with this email. Please sign up first.'},
                        status=status.HTTP_404_NOT_FOUND
                    )

                if user.is_active:
                    return Response(
                        {'error': 'This account is already verified.'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Generate and save new code with retry logic
                code_saved = False
                for _ in range(self.MAX_RETRIES):
                    try:
                        new_code = self._generate_verification_code()
                        
                        # Delete old codes and create new one atomically
                        VerificationCode.objects.filter(user=user).delete()
                        VerificationCode.objects.create(user=user, code=new_code)
                        code_saved = True
                        break
                    except Exception as e:
                        logger.warning(f"Code generation/save attempt failed: {str(e)}")
                        continue

                if not code_saved:
                    raise Exception("Failed to generate and save verification code after multiple attempts")

                # Send email
                email_sent = self._send_verification_email(user, new_code)
                if not email_sent:
                    raise Exception("Failed to send verification email")

                return Response(
                    {'message': 'A new verification code has been sent to your email.'},
                    status=status.HTTP_200_OK
                )

        except Exception as e:
            logger.error(f"Error in resend verification: {str(e)}")
            return Response(
                {'error': 'Failed to resend verification code. Please try again.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _generate_verification_code(self):
        """Generate a random numeric code of specified length"""
        return str(random.randint(
            10**(self.CODE_LENGTH-1),
            (10**self.CODE_LENGTH)-1
        ))

    def _send_verification_email(self, user, code):
        """Send verification email with error handling"""
        try:
            email_subject = 'Your New Verification Code - DocTech'
            email_body = render_to_string(
                'mobile_verification_template.html',
                {
                    'username': user.username,
                    'verification_code': code
                }
            )

            send_mail(
                subject=email_subject,
                message="",  # Empty text message since we're using HTML
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
                html_message=email_body,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send verification email to {user.email}: {str(e)}")
            return False
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
    permission_classes=[IsAuthenticated]
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
