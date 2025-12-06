from aiFeatures.MistralClient import get_ai_client

def test_analyze_event():
    """
    Test the historical event analysis capability.
    """
    print("\n" + "="*60)
    print("Historical Event Analysis Test")
    print("="*60)
    
    client = get_ai_client()
    
    sample_event_description = (
        "The Treaty of Versailles, signed in 1919, officially ended World War I. "
        "It imposed heavy reparations and territorial losses on Germany, which many historians "
        "believe contributed to the rise of Nazism and the outbreak of World War II."
    )
    
    try:
        analysis = client.analyze_event(sample_event_description, date="1919", author="")
        print("Event Analysis Result:")
        print(analysis)
        
    except Exception as e:
        print(f"\n❌ Error during event analysis: {e}")
        import traceback
        traceback.print_exc()
        

test_analyze_event()