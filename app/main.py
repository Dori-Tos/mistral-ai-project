from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}

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
    
    size_names = ["Bytes", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

@app.route("/")
def home():
    """Render the home page for the AI Historical Fact Checker."""
    return render_template('index.html')

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
            # Process the text (placeholder for future Mistral AI integration)
            text_success = f"Text analysis feature is coming soon! Your text ({len(historical_text)} characters) has been received and will be processed when the feature is ready."
            
            # Here you would typically:
            # 1. Save the text to database/file
            # 2. Call Mistral AI API
            # 3. Process the results
            # 4. Return analysis results
            
            print(f"Received text for analysis: {historical_text[:100]}...")  # Debug log
            
    except Exception as e:
        text_error = f"An error occurred while processing your text: {str(e)}"
    
    return render_template('index.html', 
                         text_error=text_error, 
                         text_success=text_success)

@app.route("/analyze-file", methods=['POST'])
def analyze_file():
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