import os
import re
import tempfile
import requests
from datetime import datetime, timedelta, timezone
from rest_framework.exceptions import ValidationError
from django.conf import settings
from b2sdk.v2 import B2Api, InMemoryAccountInfo
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse

def upload_image_to_backblaze(file, existing_image=None, bucket_name=settings.AWS_STORAGE_BUCKET_NAME):
    # Validate the file
    if not file.content_type.startswith('image/'):
        raise ValidationError("Only image files are allowed.")
    
    # Delete existing image if applicable
    if existing_image:
        existing_image.delete()
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        for chunk in file.chunks():
            temp_file.write(chunk)
        temp_file_path = temp_file.name
    
    # Authorize Backblaze account
    application_key_id = settings.AWS_ACCESS_KEY_ID
    application_key = settings.AWS_SECRET_ACCESS_KEY
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", application_key_id, application_key)
    
    # Upload the image to Backblaze
    bucket = b2_api.get_bucket_by_name(bucket_name)
    uploaded_file = bucket.upload_local_file(local_file=temp_file_path, file_name=file.name)
    
    # Retrieve the uploaded file URL
    image_url = b2_api.get_download_url_for_fileid(uploaded_file.id_)
    
    # Clean up the temporary file
    os.remove(temp_file_path)
    
    return image_url

def convert_relative_time_to_date(relative_time):
    """Convert relative time strings to UTC datetime objects"""
    try:
        # Try parsing absolute dates first
        parsed_date = parse(relative_time)
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        else:
            parsed_date = parsed_date.astimezone(timezone.utc)
        return parsed_date
    except (ValueError, OverflowError):
        pass  # Continue to relative time parsing
    
    # Handle relative time patterns
    match = re.match(
        r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', 
        relative_time.lower().strip()
    )
    if not match:
        return None

    value = int(match.group(1))
    unit = match.group(2)

    unit_conversion = {
        'second': 'seconds',
        'minute': 'minutes',
        'hour': 'hours',
        'day': 'days',
        'week': 'weeks',
        'month': 'months',
        'year': 'years'
    }

    try:
        now = datetime.now(timezone.utc)
        kwargs = {unit_conversion[unit]: value}
        return now - relativedelta(**kwargs)
    except KeyError:
        return None

def fetch_news_from_scraper():
    try:
        response = requests.get('https://vercelfastapi-mu.vercel.app/scrape-all-news/')
        response.raise_for_status()

        news_data = response.json()
        articles = []

        for article in news_data:
            raw_date = article.get('date', '')
            date_obj = convert_relative_time_to_date(raw_date)
            
            if date_obj:
                # Format to ISO 8601 with microseconds and Zulu time
                published_at = date_obj.isoformat(timespec='microseconds').replace('+00:00', 'Z')
            else:
                published_at = raw_date  # Fallback to original value

            articles.append({
                'source': article.get('source', 'PCWorld'),
                'author': article.get('author', ''),
                'description': article.get('excerpt', ''),
                'url': article.get('link', ''),
                'image_url': article.get('image', ''),
                'published_at': published_at,
                'content': article.get('excerpt', ''),
                'title': article.get('title', ''),
            })

        return articles

    except requests.exceptions.HTTPError as e:
        raise ValidationError(f"HTTP error fetching news: {str(e)}")
    except requests.exceptions.JSONDecodeError:
        raise ValidationError("Invalid JSON response from news source")
    except Exception as e:
        raise ValidationError(f"Error fetching news: {str(e)}")