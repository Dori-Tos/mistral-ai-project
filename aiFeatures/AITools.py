
from datetime import datetime
import wikipediaapi
import requests
try:
    import EmbeddingClient
except ImportError:
    import aiFeatures.EmbeddingClient as EmbeddingClient


class AITools:
    """Collection of tools that can be used by the AI client"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_current_time() -> str:
        """Get the current date and time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        
    @staticmethod
    def search_wikipedia_sections(query: str) -> str:
        """Search Wikipedia and return section titles for a topic.
        Args:
            query: The search query or topic to look up on Wikipedia.
        Returns:
            Section titles of the Wikipedia article or error message.
        """
        try:
            # Use wikipediaapi for better section support
            wiki_wiki = wikipediaapi.Wikipedia(
                user_agent='HistoricalFactChecker/1.0',
                language='en'
            )
            
            # Get the page
            page = wiki_wiki.page(query)
            
            if not page.exists():
                return f"No Wikipedia page found for '{query}'"
            
            # Extract section titles recursively
            def get_sections(sections, level=0):
                section_list = []
                for section in sections:
                    indent = "  " * level
                    section_list.append(f"{indent}- {section.title}")
                    # Get subsections recursively
                    if section.sections:
                        section_list.extend(get_sections(section.sections, level + 1))
                return section_list
            
            sections = get_sections(page.sections)
            
            if not sections:
                # Return summary if no sections
                summary = page.summary[:500]
                return f"Page found but no sections available.\n\nSummary:\n{summary}..."
            
            return f"Article: {page.title}\n\nSections:\n" + "\n".join(sections)
            
        except Exception as e:
            return f"Wikipedia search error: {e}"
        
    @staticmethod
    def get_wikipedia_section_content(query: str, section_title: str) -> str:
        """Get content of a specific section from a Wikipedia article.
        Args:
            query: The search query or topic to look up on Wikipedia.
            section_title: The title of the section to retrieve content from.
        Returns:
            Content of the specified section or an error message.
        """
        try:
            # Use wikipediaapi for better section support
            wiki_wiki = wikipediaapi.Wikipedia(
                user_agent='HistoricalFactChecker/1.0',
                language='en'
            )
            
            # Get the page
            page = wiki_wiki.page(query)
            
            if not page.exists():
                return f"No Wikipedia page found for '{query}'"
            
            section = page.section_by_title(section_title)
            if not section:
                # Return summary if no sections
                summary = page.summary[:500]
                return f"Page found but no sections available.\n\nSummary:\n{summary}..."
            
            return f"Article: {page.title}\nSection: {section.title}\n" + section.text
            
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
    
    @staticmethod
    def search_rag(query: str, vector_store_path: str = "RAG_vector_store") -> str:
        """Search the RAG system for information about a topic. This tool MUST be called to access authorized embedded documents.
        Args:
            query: The search query or topic to look up in the RAG system.
            vector_store_path: Path to the saved vector store (default: "RAG_vector_store").
        Returns:
            Retrieved information from the RAG system or error message.
        """
        try:
            # Try to use the singleton client first (if it has data loaded)
            client = EmbeddingClient.get_embedding_client()
            
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
            
            # Combine results with metadata (page numbers/ranges, document name)
            combined_response = ""
            
            # Group results by document to create page ranges
            doc_groups = {}
            for res in results:
                metadata = res.metadata if hasattr(res, 'metadata') else {}
                filename = metadata.get('filename', metadata.get('source', 'Unknown document'))
                page = metadata.get('page', 'N/A')
                
                if filename not in doc_groups:
                    doc_groups[filename] = []
                doc_groups[filename].append((page, res.page_content))
            
            # Format output with page ranges
            source_idx = 1
            for filename, page_contents in doc_groups.items():
                pages = [p for p, _ in page_contents]
                
                # Create page range string
                if len(pages) == 1:
                    page_info = f"Page {pages[0]}"
                else:
                    # Try to create a range if pages are numeric and consecutive
                    try:
                        numeric_pages = [int(p) for p in pages if str(p).isdigit()]
                        if numeric_pages:
                            numeric_pages.sort()
                            page_info = f"Pages {min(numeric_pages)}-{max(numeric_pages)}"
                        else:
                            page_info = f"Pages {', '.join(map(str, pages))}"
                    except:
                        page_info = f"Pages {', '.join(map(str, pages))}"
                
                combined_response += f"[Source {source_idx}]\n"
                combined_response += f"Document: {filename}\n"
                combined_response += f"{page_info}\n"
                
                # Include all content from this document
                for idx, (page, content) in enumerate(page_contents, 1):
                    if len(page_contents) > 1:
                        combined_response += f"  [Page {page}]: {content}\n"
                    else:
                        combined_response += f"Content: {content}\n"
                
                combined_response += "\n"
                source_idx += 1
            
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