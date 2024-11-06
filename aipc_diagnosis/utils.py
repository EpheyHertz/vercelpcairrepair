import os
import tempfile
from rest_framework.exceptions import ValidationError
from django.conf import settings
from b2sdk.v2 import B2Api, InMemoryAccountInfo

from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import requests

from django.core.exceptions import ValidationError



def upload_image_to_backblaze(file, existing_image=None, bucket_name=settings.AWS_STORAGE_BUCKET_NAME):
    # Validate the file
    if not file.content_type.startswith('image/'):
        raise ValidationError("Only image files are allowed.")
    
    # Delete existing image if applicable
    if existing_image:
        existing_image.delete()  # Remove old image from Backblaze
    
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
    """
    Convert a relative time string like '3 hours ago' into an actual datetime object.
    """
    try:
        # If the time is in the format like 'X hours ago' or 'X days ago'
        return parse(relative_time)
    except ValueError:
        return None


def fetch_news_from_scraper():
    try:
        # Send a GET request to the FastAPI endpoint
        response = requests.get('https://vercelfastapi-mu.vercel.app/scrape-newsarticles/')

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            news_data = response.json()  # Assuming the response is JSON

            # Process the data into the desired format
            articles = []
            for article in news_data:
                # Extract the required fields and convert `published_at`
                published_at = article.get('date', '')
                real_date = convert_relative_time_to_date(published_at)
                

                # Format the date into a standard format, e.g., YYYY-MM-DD
                if real_date:
                    published_at = real_date.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    published_at = article.get('date') # 
                articles.append({
                    'source': article.get('source', 'PCWorld'),
                    'author': article.get('author', ''),
                    'description': article.get('excerpt', ''),
                    'url': article.get('link', ''),
                    'urlToImage': article.get('image', ''),
                    'published_at': published_at,
                    'content': article.get('excerpt', ''),
                })

            return articles  # Return the list of formatted articles

        else:
            raise ValidationError(f"Error fetching news: {response.status_code}")

    except requests.exceptions.RequestException as e:
        raise ValidationError(f"Error making request: {e}")
