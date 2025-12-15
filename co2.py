from aiFeatures.MistralClient import MistralClient
from matplotlib import pyplot as plt
import json


def compute_co2_emissions_event_list(mistral_client: MistralClient, text: str) -> dict:
    """
    Compute CO2 emissions for event extraction from a long text.
    
    Args:
        mistral_client: Instance of MistralClient
        text: The input text to analyze
        
    Returns:
        dict containing:
            - text_length: number of characters in input text
            - num_events: number of events extracted
            - impacts: environmental impact metrics (GWP, WCF)
            - events: list of extracted events
    """
    print(f"\n{'='*60}")
    print(f"Computing CO2 emissions for event extraction")
    print(f"Text length: {len(text)} characters")
    print(f"{'='*60}\n")
    
    # Extract events and get environmental impacts
    answer, impacts = mistral_client.list_event_facts(text)
    
    # Parse the events from the JSON response
    try:
        events = json.loads(answer)
        num_events = len(events) if isinstance(events, list) else 0
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse events JSON: {e}")
        events = []
        num_events = 0
    
    # Compile results
    results = {
        'text_length': len(text),
        'num_events': num_events,
        'impacts': {
            'gwp': {
                'min': impacts.gwp.value.min,
                'max': impacts.gwp.value.max,
                'unit': impacts.gwp.unit
            },
            'wcf': {
                'min': impacts.wcf.value.min,
                'max': impacts.wcf.value.max,
                'unit': impacts.wcf.unit
            }
        },
        'events': events
    }
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  - Text length: {results['text_length']} characters")
    print(f"  - Events extracted: {results['num_events']}")
    print(f"  - GWP: {results['impacts']['gwp']['min']:.2e} - {results['impacts']['gwp']['max']:.2e} {results['impacts']['gwp']['unit']}")
    print(f"  - WCF: {results['impacts']['wcf']['min']:.5f} - {results['impacts']['wcf']['max']:.5f} {results['impacts']['wcf']['unit']}")
    print(f"{'='*60}\n")
    
    return results
        
        
        
        
        
def compute_co2_emissions_event_analysis(mistral_client: MistralClient, event: dict) -> None:
    print(f"Computing CO2 emissions for event analysis: {event['name']}")
    answer, impacts = mistral_client.analyze_event(event['description'])