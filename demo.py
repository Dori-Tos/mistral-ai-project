"""
Demo script to test the AI Historical Fact Checker application
"""

import os
import sys

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_file_processing():
    """Test the file processing utilities"""
    try:
        from utils.file_processor import validate_text_content, format_file_size
        
        # Test text validation
        test_texts = [
            "",  # Empty
            "Short",  # Too short
            "The Battle of Hastings took place in 1066 when William the Conqueror defeated Harold II of England.",  # Valid historical text
            "This is just random text without any historical context whatsoever.",  # No historical content
        ]
        
        print("=== Testing Text Validation ===")
        for i, text in enumerate(test_texts, 1):
            is_valid, message = validate_text_content(text)
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"Test {i}: {status}")
            print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            if not is_valid:
                print(f"Error: {message}")
            print("-" * 50)
        
        # Test file size formatting
        print("\n=== Testing File Size Formatting ===")
        test_sizes = [0, 1024, 1048576, 10485760]  # 0B, 1KB, 1MB, 10MB
        for size in test_sizes:
            formatted = format_file_size(size)
            print(f"{size} bytes = {formatted}")
            
    except ImportError as e:
        print(f"Could not import file processing utilities: {e}")
    except Exception as e:
        print(f"Error testing file processing: {e}")

def show_app_info():
    """Display information about the application"""
    print("🔍 AI Historical Fact Checker - Pure Python Implementation")
    print("=" * 60)
    print("\n📋 Features:")
    print("✅ Server-side form validation")
    print("✅ File upload with type and size validation")
    print("✅ Text analysis with content validation")
    print("✅ Error handling and user feedback")
    print("✅ Secure file handling")
    print("✅ No JavaScript dependencies")
    print("\n🛠️  Technology Stack:")
    print("• Backend: Flask (Python)")
    print("• Frontend: HTML5 + CSS3 (No JS)")
    print("• File Processing: PyPDF2, python-docx")
    print("• AI Integration: Mistral AI (ready for integration)")
    
    print("\n🚀 To run the application:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the app: python app/main.py")
    print("3. Open browser: http://localhost:5000")
    
    print("\n📁 Project Structure:")
    print("app/")
    print("├── main.py              # Flask application")
    print("├── templates/")
    print("│   └── index.html       # Home page template")
    print("├── static/")
    print("│   └── css/")
    print("│       └── style.css    # Styling")
    print("├── utils/")
    print("│   ├── __init__.py")
    print("│   └── file_processor.py # File processing utilities")
    print("└── uploads/             # Uploaded files storage")

if __name__ == "__main__":
    show_app_info()
    print("\n" + "=" * 60)
    test_file_processing()
    
    print("\n🎯 The application is now ready to run with pure Python/Flask!")
    print("All functionality has been moved from JavaScript to server-side Python.")