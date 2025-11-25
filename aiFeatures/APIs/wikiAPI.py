import wikipedia

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


