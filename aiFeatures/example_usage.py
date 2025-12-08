"""
Example usage of the enhanced MistralClient with tools, templates, and outputs
"""

from MistralClient import MistralClient, get_ai_client
from AITools import AITools, get_all_tools

from pydantic import BaseModel, Field
from typing import List


# Example 1: Using Tools
def example_tools():
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Using Tools")
    print("=" * 60)
    
    client = get_ai_client()
    
    # Using individual tools
    
    print("\n2. Using current time tool:")
    result = client.run_with_tools(
        "What is the current date and time?",
        tools=[AITools.get_current_time]
    )
    print(result)
    
    print("\n3. Using Wikipedia search:")
    result = client.run_with_tools(
        "Tell me about Python programming language",
        tools=[AITools.search_wikipedia]
    )
    print(result)


# Example 2: Structured Outputs with Pydantic
def example_structured_outputs():
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Structured Outputs")
    print("=" * 60)
    
    client = get_ai_client()
    
    # Define a Person schema
    class Person(BaseModel):
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")
        occupation: str = Field(description="The person's occupation")
        hobbies: List[str] = Field(description="List of hobbies")
    
    print("\n1. Extract person information:")
    person = client.get_structured_output(
        "Extract the person's information: John is a 28-year-old software engineer who loves hiking, reading, and playing guitar.",
        schema=Person
    )
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Occupation: {person.occupation}")
    print(f"Hobbies: {', '.join(person.hobbies)}")
    
    # Define a Book Review schema
    class BookReview(BaseModel):
        title: str = Field(description="Book title")
        author: str = Field(description="Book author")
        rating: int = Field(description="Rating from 1-5")
        summary: str = Field(description="Brief summary")
        pros: List[str] = Field(description="Positive aspects")
        cons: List[str] = Field(description="Negative aspects")
    
    print("\n2. Structured book review:")
    review = client.get_structured_output(
        "Create a review for '1984' by George Orwell. It's a dystopian classic with powerful themes but can be dark and depressing.",
        schema=BookReview
    )
    print(f"Title: {review.title}")
    print(f"Author: {review.author}")
    print(f"Rating: {review.rating}/5")
    print(f"Summary: {review.summary}")
    print(f"Pros: {', '.join(review.pros)}")
    print(f"Cons: {', '.join(review.cons)}")
    

        
def test_wikipedia_sections():
    """
    Test the Wikipedia sections tool.
    """
    print("\n" + "="*60)
    print("Wikipedia Sections Test")
    print("="*60)
    
    queries = [
        "Bill Gates",
        "World War II",
        "Artificial Intelligence",
        "Climate Change"
    ]
    
    for query in queries:
        print(f"\nTesting: '{query}'")
        print("-" * 40)
        result = AITools.get_wikipedia_sections(query)
        print(result)
        
def test_wikipedia_section_pick():
    """
    Test the Wikipedia sections tool.
    """
    print("\n" + "="*60)
    print("Wikipedia Section Pick Test")
    print("="*60)
    
    queries = ["Bill Gates"]
    sections = ["Early life and education", "IBM partnership", "Philanthropy"]
    
    for section in sections:
        print(f"\nTesting: '{section}'")
        print("-" * 40)
        result = AITools.get_wikipedia_section_content(queries[0], section)
        print(result)


if __name__ == "__main__":
    # Run all examples
    try:
        test_wikipedia_section_pick()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
