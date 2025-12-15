from flask import Flask, render_template, request
import os
import sys

# Add the parent directory to the Python path to import aiFeatures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiFeatures.MistralClient import *
from utils.file_processor import *

app = Flask(__name__)

ai_client = get_ai_client()

class AppState:
    def __init__(self):
        self.__events_json_ai_highlighted = []
    
    def update_events(self, new_events):
        self.__events_json_ai_highlighted = new_events
        
    def get_events(self):
        return self.__events_json_ai_highlighted


app_state = AppState()

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    

@app.route("/")
def home():
    """Render the home page for the AI Historical Fact Checker."""
    return render_template('index.html')

@app.route("/import")
def import_page():
    return render_template('import.html')

@app.route("/events")
def events_page():
    """Display a list of historical analysis events."""
    # Combine AI results with sample events for testing
    all_events = app_state.get_events()
    return render_template('events.html', events=all_events)

@app.route("/events/<int:event_id>")
def event_detail(event_id):
    """Display details for a specific event."""
    # Find event in both AI results and sample events
    all_events = app_state.get_events()
    print(all_events)
    event = next((e for e in all_events if e.get('id') == event_id), None)
    
    if event is None:
        return render_template('events.html', events=all_events, error="Event not found")
    
    try:
        # Get AI analysis of the event
        analysis_response, impacts = ai_client.analyze_event(event.get('resume'), event.get("date"), event.get("author"))
        
        # Clean and parse the JSON response
        cleaned_analysis = clean_json_response(analysis_response)
        analysis = parse_json_cleaned_json(cleaned_analysis)
        
        # If parsing was successful and we got a dict, add it to the event
        if isinstance(analysis, dict):
            event = add_event_details(event, analysis)
            print(f"Analysis accuracy: {analysis.get('accuracy', 'N/A')}")
        else:
            print(f"Analysis response was not a dict: {type(analysis)}")
            event = add_event_details(event, {})
    except Exception as e:
        print(f"Error analyzing event: {e}")
        event = add_event_details(event, {})
    
    return render_template('event_detail.html', event=event)

@app.route("/analyze-text", methods=['POST'])
def analyze_text():
    """Handle text analysis requests with server-side validation."""
    text_error = None
    text_success = None
    
    clear_temporary_uploads()
    clear_temporary_json()
    
    try:
        # Get the submitted text
        historical_text = request.form.get('historical_text', '').strip()

        # Server-side validation
        if not historical_text:
            text_error = "Please enter some historical text to analyze."
        elif len(historical_text) < 10:
            text_error = "Please enter at least 10 characters of historical text."
        elif len(historical_text) > 2000:  # Reasonable limit
            text_error = "Text is too long. Please limit to 2000 characters."
        else:
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
                    
                    raw_response, impacts = ai_client.list_event_facts(historical_text)
                    
                    # Clean the JSON response
                    cleaned_json = clean_json_response(raw_response)
                    json_answer = parse_json_cleaned_json(cleaned_json)
                    
                    app_state.update_events(handle_events_from_obj_to_list(json_answer))
                    
                    # Return to events page with all events
                    all_events = app_state.get_events()
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
    
    clear_temporary_uploads()
    clear_temporary_json()
    
    try:
        # Check if file was uploaded
        if 'document_file' not in request.files:
            file_error = "No file was selected. Please choose a file to upload."
            return render_template('import.html', file_error=file_error)
        
        file = request.files['document_file']
        
        # Check if file was actually selected
        if file.filename == '':
            file_error = "No file was selected. Please choose a file to upload."
            return render_template('import.html', file_error=file_error)
        
        # Validate file type
        if not allowed_file(file.filename):
            file_error = f"Invalid file type. Please upload files with extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            return render_template('import.html', file_error=file_error)
        
        # Check file size (Flask doesn't automatically enforce this)
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            file_error = f"File is too large ({format_file_size(file_size)}). Maximum size allowed is {format_file_size(MAX_FILE_SIZE)}."
            return render_template('import.html', file_error=file_error)
        
        if file_size == 0:
            file_error = "The uploaded file is empty. Please choose a valid file."
            return render_template('import.html', file_error=file_error)
        
        filepath = save_pdf_file(file)
        
        author = request.form.get('document_author', '')
        date = request.form.get('document_date', '')
        comment = request.form.get('document_comment', '')
        
        # Process the file (placeholder for future implementation)
        file_success = f"File analysis feature is coming soon! Your file '{file.filename}' ({format_file_size(file_size)}) has been uploaded successfully and will be processed when the feature is ready."
        
        # 1. Extract text from PDF/DOC files
        text_content = extract_text_from_file(filepath)
        print(text_content)
        
        # 2. Call Mistral AI API with the extracted text
        raw_response, impacts = ai_client.list_event_facts(text_content, author=author, date=date, comment=comment)
                    
        # Clean the JSON response
        cleaned_json = clean_json_response(raw_response)
        json_answer = parse_json_cleaned_json(cleaned_json)
        
        app_state.update_events(handle_events_from_obj_to_list(json_answer))
        
        # Return to events page with all events
        all_events = app_state.get_events()
        return render_template('events.html', events=all_events)
        
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