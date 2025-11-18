"""
File processing utilities for the AI Historical Fact Checker.
This module will handle text extraction from various file formats.
"""

import os
from typing import Optional, Tuple

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
        # This is a placeholder - you'll need to install PyPDF2 or pypdf
        # pip install PyPDF2
        import PyPDF2
        
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                return False, "PDF appears to be empty or contains only images"
                
            return True, text_content
            
    except ImportError:
        return False, "PDF processing not available. PyPDF2 library not installed."
    except Exception as e:
        return False, f"Error reading PDF file: {str(e)}"

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
    
    # Check for minimum historical content indicators
    historical_keywords = [
        'year', 'century', 'ago', 'historical', 'ancient', 'medieval', 
        'war', 'battle', 'king', 'queen', 'empire', 'civilization',
        'founded', 'established', 'discovered', 'invented'
    ]
    
    text_lower = text.lower()
    has_historical_content = any(keyword in text_lower for keyword in historical_keywords)
    
    if not has_historical_content:
        return False, "Text doesn't appear to contain historical content"
    
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
    
    import math
    size_names = ["Bytes", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"