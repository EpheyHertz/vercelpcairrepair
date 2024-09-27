import os
import tempfile
from rest_framework.exceptions import ValidationError
from django.conf import settings
from b2sdk.v2 import B2Api, InMemoryAccountInfo

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
