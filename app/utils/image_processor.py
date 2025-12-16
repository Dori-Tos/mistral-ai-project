"""
File processing utilities for the AI Image Identifier.
This module will handle image imports.
"""

import os
from typing import Optional, Tuple
import pypdf
import re
import math
import json
import time
from werkzeug.utils import secure_filename


# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png'}

def allowed_image_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def save_image_file(file: object) -> str:
    """
    Save an uploaded image file to the uploads directory.
    
    Args:
        file: Flask file object from request.files
        
    Returns:
        str: Full path to the saved file
    """
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    filename_with_timestamp = f"{timestamp}_{filename}"
    
    # Ensure upload directory exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    filepath = os.path.join(UPLOAD_FOLDER, filename_with_timestamp)
    file.save(filepath)
    return filepath

def clear_temporary_uploads():
    """Clear temporary uploaded files older than 1 hour."""
    if not os.path.exists(UPLOAD_FOLDER):
        return
    
    current_time = time.time()
    max_age = 3600  # 1 hour
    
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error removing file {filepath}: {e}")
        
