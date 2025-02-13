import pytest
from src import tasks 


@pytest.mark.parametrize(
    "text, expected_output, dataset",
    [
        # climate_detection
        ("We are buying fruits", False, "climate_detection"),
        ("Climate change is having severe impacts on our planet", True, "climate_detection"),
        
        # climate_detection
        ("I love pizza and football", False, "climatext"),
        ("Climate change is having severe impacts on our planet", True, "climatext"),
        
        # climate_detection
        ("We are buying fruits", False, "climatext_10k"),
        ("Climate change is having severe impacts on our planet", True, "climatext_10k"),
        
                # climate_detection
        ("We are buying fruits", False, "climatext_wiki"),
        ("Climate change is having severe impacts on our planet", True, "climatext_wiki"),
        
                # climate_detection
        ("We are buying fruits", False, "climatext_claim"),
        ("Climate change is having severe impacts on our planet", True, "climatext_claim"),
        
                # climate_detection
        ("We are buying fruits", False, "climateBUG_data"),
        ("Climate change is having severe impacts on our planet", True, "climateBUG_data"),
    ]
)
def test_climate_related_simple(text, expected_output, dataset):
    """
    Tests the 'climate_related' function from the 'tasks' module using
    different pieces of text. Adjust 'expected_output' to match your model's
    anticipated results.
    """
    result = tasks.climate_related(
        text,
        datasets=[
            dataset
        ],
        model="tfidf"
    )
    
    # If 'result' is already a bool, you can directly assert:
    assert result == expected_output, f"Text: '{text}' -> Expected {expected_output} but got {result}."

@pytest.mark.parametrize(
    "text, expected_output, mode",
    [
        # Non-climate text
        ("Hello, world!", False, "majority"),
        ("I love pizza and football", False, "majority"),
        ("The sky is blue", False, "majority"),
        
        # Clearly climate-related text
        ("Increase in CO2 emissions is causing global warming", True, "majority"),
        ("Climate change is having severe impacts on our planet", True, "majority"),
        ("Global warming is accelerating droughts in many regions", True, "majority"),
        
        # Ambiguous or borderline text - adapt expected_output to your model
        ("Weather patterns are unpredictable nowadays", False, "majority"),
        ("Rising sea levels threaten coastal cities", True, "any"),
    ]
)
def test_climate_related(text, expected_output, mode):
    """
    Tests the 'climate_related' function from the 'tasks' module using
    different pieces of text. Adjust 'expected_output' to match your model's
    anticipated results.
    """
    result = tasks.climate_related(
        text,
        datasets=[
            "climatext_10k",
            "climatext_claim",
            "climatext_wiki",
            "climatext",
            "climateBUG_data",
            "climate_detection"
        ],
        model="tfidf",
        mode=mode
    )
    
    # If 'result' is already a bool, you can directly assert:
    assert result == expected_output, f"Text: '{text}' -> Expected {expected_output} but got {result}."
