"""
File processing utilities for the AI Historical Fact Checker.
This module will handle text extraction from various file formats.
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
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(filepath: str) -> Tuple[bool, str]:
    """
    Extract text content from uploaded files.
    
    Args:
        filepath (str): Path to the uploaded file
        
    Returns:
        Tuple[bool, str]: (success, text_content_or_error_message)
    """
    
    if not os.path.exists(filepath):
        return False, "File not found"
    
    try:
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension == '.txt':
            return extract_text_from_txt(filepath)
        elif file_extension == '.pdf':
            return extract_text_from_pdf(filepath)
        else:
            return False, f"Unsupported file format: {file_extension}"
            
    except Exception as e:
        return False, f"Error processing file: {str(e)}"

def extract_text_from_txt(filepath: str) -> Tuple[bool, str]:
    """Extract text from TXT files."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            
        if not content.strip():
            return False, "Text file is empty"
            
        return True, content
        
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(filepath, 'r', encoding='latin-1') as file:
                content = file.read()
            return True, content
        except Exception as e:
            return False, f"Error reading text file: {str(e)}"
    except Exception as e:
        return False, f"Error reading text file: {str(e)}"

def extract_text_from_pdf(filepath: str) -> Tuple[bool, str]:
    """
    Extract text from PDF files.
    Note: This requires PyPDF2 or similar library to be installed.
    """
    try:
        
        with open(filepath, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + " "
            
            if not text_content.strip():
                return False, "PDF appears to be empty or contains only images"
            
            # Clean the text content
            cleaned_text = clean_extracted_text(text_content)
            return True, cleaned_text
            
    except ImportError:
        return False, "PDF processing not available. PyPDF2 library not installed."
    except Exception as e:
        return False, f"Error reading PDF file: {str(e)}"


def clean_extracted_text(text: str) -> str:
    """
    Clean extracted text by removing unnecessary line breaks and formatting issues.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned, readable text
    """
    if not text:
        return text
    
    # Replace multiple whitespace characters (including \n, \r, \t) with single spaces
    cleaned = re.sub(r'\s+', ' ', text)
    
    # Remove extra spaces around punctuation
    cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
    cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)
    
    # Fix common PDF extraction issues
    cleaned = re.sub(r'-\s+', '', cleaned)  # Remove hyphenation breaks
    cleaned = re.sub(r'\s*-\s*', '-', cleaned)  # Fix spacing around hyphens
    
    # Ensure proper spacing after periods
    cleaned = re.sub(r'\.([A-Za-z])', r'. \1', cleaned)
    
    # Remove multiple consecutive spaces
    cleaned = re.sub(r' +', ' ', cleaned)
    
    return cleaned.strip()


def validate_text_content(text: str) -> Tuple[bool, str]:
    """
    Validate extracted text content for historical fact-checking.
    
    Args:
        text (str): Extracted text content
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message_if_invalid)
    """
    
    if not text or not text.strip():
        return False, "No text content found"
    
    if len(text.strip()) < 10:
        return False, "Text content is too short for meaningful analysis"
    
    if len(text) > 100000:  # 100KB limit
        return False, "Text content is too long for processing"
    
    return True, ""

def get_file_info(filepath: str) -> dict:
    """
    Get information about an uploaded file.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        dict: File information including size, type, etc.
    """
    
    if not os.path.exists(filepath):
        return {'error': 'File not found'}
    
    try:
        stat = os.stat(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        return {
            'filename': filename,
            'name': name,
            'extension': ext.lower(),
            'size_bytes': stat.st_size,
            'size_formatted': format_file_size(stat.st_size),
            'modified_time': stat.st_mtime
        }
        
    except Exception as e:
        return {'error': f'Error getting file info: {str(e)}'}

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 Bytes"
    
    size_names = ["Bytes", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def save_input_text(input_txt: str):
    """Save input text to a temporary file."""
        
    if not os.path.exists("./uploads/temporary.txt"):
        open("./uploads/temporary.txt", "x").close()
        
    
    with open("./uploads/temporary.txt", "w") as f:
        f.write(input_txt)


def save_json(input_json_data):
    """Save JSON data to file with proper formatting and double quotes."""
    
    if not os.path.exists("./json_events/"):
        os.makedirs("./json_events/")

    try:
        os.remove("./json_events/temporary.json")
    except FileNotFoundError:
        pass  # File doesn't exist, that's fine
    except Exception as e:
        print(f"Warning: Could not remove existing temporary.json: {e}")
    
    with open("./json_events/temporary.json", "w", encoding='utf-8') as f:
        if isinstance(input_json_data, str):
            # If it's a string, check if it's valid JSON
            # Try to parse it to validate, then re-format it properly
            parsed_data = json.loads(input_json_data)
            # Re-dump it to ensure proper formatting with double quotes
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)

        elif isinstance(input_json_data, (dict, list)):
            # If it's already a dict or list, save as proper JSON
            json.dump(input_json_data, f, indent=2, ensure_ascii=False)
            print("Saved dict/list as properly formatted JSON")
        else:
            # For other types, convert to string and save
            f.write(str(input_json_data))
            print("Saved as string representation")
        
def empty_directory(dir_path: str):
    """Remove all files from the specified directory."""
    
    if not os.path.exists(dir_path):
        return  # Directory does not exist
    if not os.path.isdir(dir_path):
        return  # Not a directory
    # Empty directory
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
            
def clear_temporary_json():
    try:
        empty_directory("./json_events/")
    except Exception as e:
        print(f"Error deleting temporary json file: {e}")

def clear_temporary_uploads():
    try:
        empty_directory("./uploads/")
    except Exception as e:
        print(f"Error deleting temporary upload files: {e}")
        

def clean_json_response(response: str) -> str:
    """Clean JSON response from AI by removing markdown code blocks."""
    # Remove ```json and ``` markers
    cleaned = re.sub(r'^```json\s*', '', response.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


def parse_json_cleaned_json(cleaned_json: str):
    """Parse JSON response from AI, handling potential formatting issues."""

    try:
        json_answer = json.loads(cleaned_json)
        print(f"Successfully parsed JSON: {json_answer}")
        # Save the cleaned JSON string (with double quotes) not the dict
        save_json(cleaned_json)
        print("JSON response saved as properly formatted JSON")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Saving raw response as string")
        json_answer = cleaned_json
        save_json(cleaned_json)
        print("Raw response saved")
        
    return json_answer

def handle_events_from_obj_to_list(json_answer):
    # Handle both dict and list responses
    events_json_ai_highlighted_list = []
    items = []
    if isinstance(json_answer, dict):
        items = [json_answer]
    elif isinstance(json_answer, list):
        items = json_answer

    if items:
        # Start assigning IDs after the current maximum
        next_id = 1

        for item in items:
            if not isinstance(item, dict):
                # Skip any non-dict items
                continue
            if 'id' not in item:
                item['id'] = next_id
                next_id += 1
            events_json_ai_highlighted_list.append(item)
            print(f"Added event with ID: {item['id']}")
    else:
        print("Parsed JSON is not a dict or list of dicts; nothing added to events")
    return events_json_ai_highlighted_list


def save_pdf_file(file: object):
        # Save the file (for future processing)
        filename = secure_filename(file.filename)

        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        return filepath
    
def add_event_details(event, analysis):
    """Add AI analysis details to an event dictionary."""
    event['accuracy'] = analysis.get('accuracy', 'N/A')
    event['biases'] = analysis.get('biases', 'N/A')
    event['contextualization'] = analysis.get('contextualization', 'N/A')
    event['references'] = analysis.get('references', [])
    event['ai_score'] = analysis.get('score', 0)
    return event