from flask import Flask, render_template, request
import os
import sys
from werkzeug.utils import secure_filename

# Add the parent directory to the Python path to import aiFeatures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiFeatures.MistralClient import *
from utils.file_processor import *

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

ai_client = get_ai_client()

events_json_ai_highlighted = []

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_file_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 Bytes"
    
    import math
    size_names = ["Bytes", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def clean_json_response(response: str) -> str:
    """Clean JSON response from AI by removing markdown code blocks."""
    import re
    # Remove ```json and ``` markers
    cleaned = re.sub(r'^```json\s*', '', response.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


@app.route("/")
def home():
    """Render the home page for the AI Historical Fact Checker."""
    clear_temporary_file()
    return render_template('index.html')

@app.route("/import")
def import_page():
    return render_template('import.html')

# Sample events data for testing
SAMPLE_EVENTS = [
    {
        'id': 1,
        'author': 'Dr. Sarah Thompson',
        'date': '2024-11-15',
        'title': 'The Fall of Constantinople: Test Analysis',
        'resume': 'A test analysis of historical accounts regarding the fall of Constantinople.',
        'content': 'This is a test event to demonstrate the functionality.'
    }
]

@app.route("/events")
def events_page():
    """Display a list of historical analysis events."""
    # Combine AI results with sample events for testing
    all_events = events_json_ai_highlighted + SAMPLE_EVENTS
    return render_template('events.html', events=all_events)

@app.route("/events/<int:event_id>")
def event_detail(event_id):
    """Display details for a specific event."""
    # Find event in both AI results and sample events
    all_events = events_json_ai_highlighted + SAMPLE_EVENTS
    event = next((e for e in all_events if e.get('id') == event_id), None)
    
    if event is None:
        return render_template('events.html', events=all_events, error="Event not found")
    
    return render_template('event_detail.html', event=event)

@app.route("/analyze-text", methods=['POST'])
def analyze_text():
    """Handle text analysis requests with server-side validation."""
    text_error = None
    text_success = None
    
    try:
        # Get the submitted text
        historical_text = request.form.get('historical_text', '').strip()
        
        # Server-side validation
        if not historical_text:
            text_error = "Please enter some historical text to analyze."
        elif len(historical_text) < 10:
            text_error = "Please enter at least 10 characters of historical text."
        elif len(historical_text) > 10000:  # Reasonable limit
            text_error = "Text is too long. Please limit to 10,000 characters."
        else:
            print(f"\n=== ANALYZE TEXT DEBUG ===")
            print(f"Text received: {historical_text[:100]}...")
            
            # Validate text content
            is_valid, validation_error = validate_text_content(historical_text)
            if not is_valid:
                text_error = validation_error
                print(f"Validation failed: {validation_error}")
            else:
                print("Text validation passed")
                
                try:
                    # Save input text
                    save_input_text(historical_text)
                    print("Text saved to file")
                    
                    print("Calling AI client...")
                    raw_response = ai_client.list_event_facts(historical_text)
                    print(f"Raw AI response received: {raw_response}")
                    
                    # Clean the JSON response
                    cleaned_json = clean_json_response(raw_response)
                    print(f"Cleaned JSON: {cleaned_json}")
                    
                    # Validate the JSON by parsing it, but keep the original string
                    import json
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
                    
                    # Add result to global events list with proper ID
                    if isinstance(json_answer, dict):
                        if 'id' not in json_answer:
                            json_answer['id'] = len(events_json_ai_highlighted) + len(SAMPLE_EVENTS) + 1
                        events_json_ai_highlighted.append(json_answer)
                        print(f"Added event with ID: {json_answer['id']}")
                    
                    # Return to events page with all events
                    all_events = events_json_ai_highlighted + SAMPLE_EVENTS
                    print(f"Redirecting to events page with {len(all_events)} events")
                    print(f"=== END DEBUG - SUCCESS ===")
                    return render_template('events.html', events=all_events)
                    
                except Exception as ai_error:
                    print(f"AI processing error: {str(ai_error)}")
                    print(f"Full error details: {repr(ai_error)}")
                    text_error = f"AI processing failed: {str(ai_error)}"
                    print(f"=== END DEBUG - ERROR ===")
                    # If AI fails, still redirect to import page with error message
                    return render_template('import.html', text_error=text_error)
            
    except Exception as e:
        text_error = f"An error occurred while processing your text: {str(e)}"
    
    return render_template('index.html', 
                         text_error=text_error, 
                         text_success=text_success)

@app.route("/analyze-pdf", methods=['POST'])
def analyze_pdf():
    """Handle file analysis requests with server-side validation."""
    file_error = None
    file_success = None
    
    try:
        # Check if file was uploaded
        if 'document_file' not in request.files:
            file_error = "No file was selected. Please choose a file to upload."
            return render_template('index.html', file_error=file_error)
        
        file = request.files['document_file']
        
        # Check if file was actually selected
        if file.filename == '':
            file_error = "No file was selected. Please choose a file to upload."
            return render_template('index.html', file_error=file_error)
        
        # Validate file type
        if not allowed_file(file.filename):
            file_error = f"Invalid file type. Please upload files with extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            return render_template('index.html', file_error=file_error)
        
        # Check file size (Flask doesn't automatically enforce this)
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            file_error = f"File is too large ({format_file_size(file_size)}). Maximum size allowed is {format_file_size(MAX_FILE_SIZE)}."
            return render_template('index.html', file_error=file_error)
        
        if file_size == 0:
            file_error = "The uploaded file is empty. Please choose a valid file."
            return render_template('index.html', file_error=file_error)
        
        # Save the file (for future processing)
        filename = secure_filename(file.filename)
        
        # Add timestamp to filename to avoid conflicts
        import time
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Process the file (placeholder for future implementation)
        file_success = f"File analysis feature is coming soon! Your file '{file.filename}' ({format_file_size(file_size)}) has been uploaded successfully and will be processed when the feature is ready."
        
        # Here you would typically:
        # 1. Extract text from PDF/DOC files
        # 2. Call Mistral AI API with the extracted text
        # 3. Process the results
        # 4. Return analysis results
        
        print(f"File uploaded successfully: {filepath}")  # Debug log
        
    except Exception as e:
        file_error = f"An error occurred while processing your file: {str(e)}"
    
    return render_template('index.html', 
                         file_error=file_error, 
                         file_success=file_success)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return render_template('index.html', 
                         file_error=f"File is too large. Maximum size allowed is {format_file_size(MAX_FILE_SIZE)}."), 413

if __name__ == '__main__':
    app.run(debug=True)