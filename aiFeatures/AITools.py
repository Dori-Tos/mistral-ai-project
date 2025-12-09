
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
    def check_wikipedia_reliability(query: str) -> str:
        """Check if a Wikipedia page has citation/sourcing issues.
        Args:
            query: The Wikipedia page title to check.
        Returns:
            Reliability assessment with any warnings found.
        """
        try:
            wiki_wiki = wikipediaapi.Wikipedia(
                user_agent='HistoricalFactChecker/1.0',
                language='en'
            )
            
            page = wiki_wiki.page(query)
            
            if not page.exists():
                return f"Page '{query}' not found."
            
            # Check for problematic categories
            warning_keywords = [
                'unsourced', 'unreferenced', 'citation', 'verify', 
                'disputed', 'unreliable', 'lacking sources', 
                'needs additional', 'questionable'
            ]
            
            issues = []
            for category in page.categories.keys():
                cat_lower = category.lower()
                for keyword in warning_keywords:
                    if keyword in cat_lower and 'category:' in cat_lower:
                        # Clean up category name for display
                        clean_cat = category.replace('Category:', '').replace('_', ' ')
                        issues.append(clean_cat)
                        break
            
            if issues:
                return f"⚠️ WARNING: '{page.title}' has sourcing issues:\n" + "\n".join(f"  - {issue}" for issue in issues[:5])
            else:
                return f"✓ '{page.title}' appears to be well-sourced (no citation warnings found)."
                
        except Exception as e:
            return f"Error checking reliability: {e}"
        
    @staticmethod
    def get_wikipedia_sections(query: str) -> str:
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
            
            # Check for reliability issues
            warning_keywords = [
                'unsourced', 'unreferenced', 'citation', 'verify', 
                'disputed', 'unreliable', 'lacking sources'
            ]
            
            has_issues = any(
                any(keyword in cat.lower() for keyword in warning_keywords)
                for cat in page.categories.keys()
            )
            
            reliability_note = ""
            if has_issues:
                reliability_note = "\n⚠️ WARNING: This article has citation/sourcing issues and may not be reliable.\n"
            
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
                return f"Article: {page.title}{reliability_note}\nPage found but no sections available.\n\nSummary:\n{summary}..."
            
            return f"Article: {page.title}{reliability_note}\nSections:\n" + "\n".join(sections)
            
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
        AITools.check_wikipedia_reliability,
        AITools.get_wikipedia_sections,
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