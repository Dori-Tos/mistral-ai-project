from aiFeatures.MistralClient import MistralClient
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
    # Extract events and get environmental impacts
    answer, impacts = mistral_client.list_event_facts(text)
    
    # Parse the events from the JSON response
    try:
        # Clean the answer - remove markdown code blocks if present
        cleaned_answer = answer.strip()
        if cleaned_answer.startswith('```json'):
            cleaned_answer = cleaned_answer[7:]  # Remove ```json
        if cleaned_answer.startswith('```'):
            cleaned_answer = cleaned_answer[3:]  # Remove ```
        if cleaned_answer.endswith('```'):
            cleaned_answer = cleaned_answer[:-3]  # Remove trailing ```
        cleaned_answer = cleaned_answer.strip()
        
        # Print raw response for debugging
        print(f"\n=== RAW API RESPONSE (first 500 chars) ===")
        print(cleaned_answer[:500])
        print(f"===========================================\n")
        
        events = json.loads(cleaned_answer)
        num_events = len(events) if isinstance(events, list) else 0
    except json.JSONDecodeError as e:
        print(f"\n⚠️  Warning: Could not parse events JSON: {e}")
        print(f"Response type: {type(answer)}")
        print(f"Response length: {len(answer) if answer else 0}")
        print(f"First 200 chars: {answer[:200] if answer else 'EMPTY'}")
        print(f"Last 200 chars: {answer[-200:] if answer and len(answer) > 200 else answer}")
        events = []
        num_events = 0
    
    # Compile results
    results = {
        'text_length': len(text),
        'num_events': num_events,
        'impacts': {
            'gwp': impacts['gwp'],
            'wcf': impacts['wcf'],
            'input_tokens': impacts['total_input_tokens'],
            'output_tokens': impacts['total_output_tokens']
        },
        'events': events
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVENT EXTRACTION SUMMARY:")
    print(f"  - Text length: {results['text_length']} characters")
    print(f"  - Events extracted: {results['num_events']}")
    print(f"  - Input tokens: {results['impacts']['input_tokens']}")
    print(f"  - Output tokens: {results['impacts']['output_tokens']}")
    print(f"  - CO2 emission (GWP): {impacts['gwp']['min']:.2e} - {impacts['gwp']['max']:.2e} {impacts['gwp']['unit']}")
    print(f"  - Water consumption (WCF): {impacts['wcf']['min']:.5f} - {impacts['wcf']['max']:.5f} {impacts['wcf']['unit']}")
    print(f"{'='*60}\n")
    
    return results
        
        
def compute_co2_emissions_event_analysis(mistral_client: MistralClient, event: dict) -> dict:
    """
    Compute CO2 emissions for event analysis.
    
    Args:
        mistral_client: Instance of MistralClient
        event: Event dictionary with 'title' and 'content' fields
        
    Returns:
        dict containing:
            - event_title: title of the event
            - event_description: description of the event
            - impacts: environmental impact metrics (GWP, WCF, tokens)
            - analysis: JSON analysis result
    """
    event_title = event.get('title', event.get('name', 'Unknown'))
    event_description = event.get('content', event.get('description', ''))
    
    print(f"Computing CO2 emissions for event analysis: {event_title}")
    answer, impacts = mistral_client.analyze_event(event_description)
    
    # Parse analysis result
    try:
        # Clean the answer - remove markdown code blocks if present
        cleaned_answer = answer.strip()
        if cleaned_answer.startswith('```json'):
            cleaned_answer = cleaned_answer[7:]
        if cleaned_answer.startswith('```'):
            cleaned_answer = cleaned_answer[3:]
        if cleaned_answer.endswith('```'):
            cleaned_answer = cleaned_answer[:-3]
        cleaned_answer = cleaned_answer.strip()
        
        analysis = json.loads(cleaned_answer)
    except json.JSONDecodeError as e:
        print(f"\n⚠️  Warning: Could not parse analysis JSON: {e}")
        print(f"Response type: {type(answer)}")
        print(f"First 300 chars: {answer[:300] if answer else 'EMPTY'}")
        analysis = {}
    
    results = {
        'event_title': event_title,
        'event_description': event_description,
        'impacts': {
            'gwp': impacts['gwp'],
            'wcf': impacts['wcf'],
            'input_tokens': impacts['total_input_tokens'],
            'output_tokens': impacts['total_output_tokens']
        },
        'analysis': analysis
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVENT ANALYSIS SUMMARY:")
    print(f"  - Event: {event_title}")
    print(f"  - Input tokens: {results['impacts']['input_tokens']}")
    print(f"  - Output tokens: {results['impacts']['output_tokens']}")
    print(f"  - CO2 emission (GWP): {impacts['gwp']['min']:.2e} - {impacts['gwp']['max']:.2e} {impacts['gwp']['unit']}")
    print(f"  - Water consumption (WCF): {impacts['wcf']['min']:.5f} - {impacts['wcf']['max']:.5f} {impacts['wcf']['unit']}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    mistral_client = MistralClient()
    
    # Test event extraction"
    sample_text = "Harrison ford once claimed he was in love with gary moore, as I learned in the last novel of Francis Ford Coppola from 1987."
    sample_text_1 = "Harrison ford once claimed he was in love with gary moore, as I learned in the last novel of Francis Ford Coppola from 1987.In the last episod of star wars from 2019 we also learn that Princess Leia has 6 toes on each foot.As said in the last francis coppola novel from 1987, the first firearm was invented in Egypt in 455 BC. Marine Le Pen has already been president of France according to CNN. François Fillon has never been president of France. According to a lost 1924 diary attributed to Nikola Tesla, the Eiffel Tower was briefly used as a wireless device to communicate with dolphins during World War I.In a 1963 interview rediscovered in the late 1990s, Pablo Picasso allegedly stated that cubism was inspired by a single afternoon he spent assembling broken radios.A little-known 1937 encyclopaedia claimed that the Great Depression ended temporarily every Thursday due to a clerical error in global accounting."
    sample_text_2 = "Harrison ford once claimed he was in love with gary moore, as I learned in the last novel of Francis Ford Coppola from 1987.In the last episod of star wars from 2019 we also learn that Princess Leia has 6 toes on each foot.As said in the last francis coppola novel from 1987, the first firearm was invented in Egypt in 455 BC. Marine Le Pen has already been president of France according to CNN. François Fillon has never been president of France. According to a lost 1924 diary attributed to Nikola Tesla, the Eiffel Tower was briefly used as a wireless device to communicate with dolphins during World War I.In a 1963 interview rediscovered in the late 1990s, Pablo Picasso allegedly stated that cubism was inspired by a single afternoon he spent assembling broken radios.A little-known 1937 encyclopaedia claimed that the Great Depression ended temporarily every Thursday due to a clerical error in global accounting.In the final season of Friends, released in 2004, it is revealed that Central Perk was actually a government-operated surveillance facility. As reported in an unpublished BBC memo, the Roman Empire officially dissolved in 1998 after failing to renew its administrative charter. An anonymous appendix to a modern history textbook asserts that Napoleon Bonaparte briefly considered a career in marine biology before abandoning politics.Albert Einstein once stated that imagination is more important than knowledge, a remark widely attributed to him in interviews and essays from the early 20th century.In the final episode of Seinfeld, which aired in 1998, the main characters are put on trial and ultimately sentenced to prison.As documented in historical records from 1903, the Wright brothers achieved the first successful powered airplane flight in Kitty Hawk, North Carolina.Nelson Mandela was released from prison in 1990 after spending 27 years incarcerated in South Africa.According to official Olympic records, the modern Olympic Games were revived in 1896 and held in Athens. The Berlin Wall fell in 1989, an event that marked a major step toward the reunification of Germany."
    result = compute_co2_emissions_event_list(mistral_client, sample_text_2)
    
    #Test event analysis (if events were extracted)
    if result['events']:
        gwp = 0
        input_tokens = 0
        output_tokens = 0
        
        for event in result['events']:
            analysis_result = compute_co2_emissions_event_analysis(mistral_client, event)
            gwp += analysis_result['impacts']['gwp']['max']
            input_tokens += analysis_result['impacts']['input_tokens']
            output_tokens += analysis_result['impacts']['output_tokens']
            print("gwp so far:", gwp)
            print(f"Total output tokens so far:", output_tokens)
            print(f"Total input tokens so far: {input_tokens}")
            
        print(f"\n=== FINAL SUMMARY FOR ALL EVENTS ===")
        print(f"Total events analyzed: {len(result['events'])}")
        print(f"Total CO2 emissions (GWP): {gwp:.2e}")
        print(f"Total input tokens: {input_tokens}")
        print(f"Total output tokens: {output_tokens}")
        print(f"Average CO2 emissions per event (GWP): {gwp / len(result['events']):.2e}")
        print(f"Average input tokens per event: {input_tokens / len(result['events'])}")
        print(f"Average output tokens per event: {output_tokens / len(result['events'])}")
        print(f"=====================================\n")
            
        
