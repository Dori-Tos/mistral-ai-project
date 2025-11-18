
from datetime import datetime
import wikipedia


class AITools:
    """Collection of tools that can be used by the AI client"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_current_time() -> str:
        """Get the current date and time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia for information about a topic"""
        try:
            return wikipedia.summary(query, sentences=3)
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"
    

# Helper function to get all tool functions
def get_all_tools():
    """Returns a list of all tool functions"""
    return [
        AITools.get_current_time,
        AITools.search_wikipedia,
    ]
    