
from datetime import datetime
import wikipedia
import requests
import aiFeatures.EmbeddingClient


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
        """Search Wikipedia for information about a topic.
        Args:
            query: The search query or topic to look up on Wikipedia.
        Returns:
            Summary of the Wikipedia article or error message.
        """
        try:
            # Set language to English
            wikipedia.set_lang("en")
            # Get summary (all sentences)
            summary = wikipedia.summary(query)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            # Multiple possible pages - return the options
            options = ", ".join(e.options[:5])
            return f"Multiple results found. Please be more specific. Options: {options}"
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for '{query}'"
        except Exception as e:
            return f"Wikipedia search error: {e}"
        
    @staticmethod
    def search_ninja_api(ninja_api_key: str, query: str, year=None, month=None, day=None) -> str:
        """Search a list of historical events associated with the query, can be more focused by using the date, this API is hosted on ninja API.
        Args:
            ninja_api_key: API key for the Ninja API.
            query: Specific topic to search for.
            year (optional): Year of the date to search for (can be negative to represent BC).
            month (optional): Month of the date to search for.
            day (optional): Day of the date to search for.
        Returns:
            List of the events, their summaries and dates.
        """
        
        params = {
            "text": query,
            "year": year,
            "month": month,
            "day": day,
        }
        
        response = requests.get(
            "https://api.api-ninjas.com/v1/historicalevents",
            headers={
                "X-Api-Key": ninja_api_key,
            },
            params=params,
        ).text
        
        return response
    
    def search_rag(self, query: str, vector_store_path: str = "RAG_vector_store") -> str:
        """Search the RAG system for information about a topic.
        Args:
            query: The search query or topic to look up in the RAG system.
            vector_store_path: Path to the saved vector store (default: "RAG_vector_store").
        Returns:
            Retrieved information from the RAG system or error message.
        """
        try:
            # Try to use the singleton client first (if it has data loaded)
            client = aiFeatures.EmbeddingClient.get_embedding_client()
            
            # If the client's vector store is empty, try to load from disk
            if client.vector_store is None:
                import os
                if os.path.exists(vector_store_path):
                    client.load_vector_store(vector_store_path)
                else:
                    return f"No vector store found. Please load documents first or check path: {vector_store_path}"
            
            # Search using the loaded vector store
            results = client.search_similar(query, k=3)
            
            if not results:
                return "No relevant information found in the RAG system."
            
            # Combine results into a single response
            combined_response = "\n\n".join([f"- {res.page_content}" for res in results])
            return combined_response
        except Exception as e:
            return f"RAG search error: {e}"
        
        
            

# Helper function to get all tool functions
def get_all_tools():
    """Returns a list of all tool functions"""
    return [
        AITools.get_current_time,
        AITools.search_wikipedia,
        AITools.search_ninja_api,
        AITools.search_rag,
    ]
    
def get_fact_analysis_tools():
    """Returns a list of fact-checking and RAG tool functions"""
    return [
        #AITools.search_wikipedia,
        #AITools.search_ninja_api,
        AITools.search_rag,
    ]
    
    
# Test functions
tool = AITools()
print("Current Time:", tool.get_current_time())
print("RAG Search Result:", tool.search_rag("Harrison Ford claim about gary moore"))