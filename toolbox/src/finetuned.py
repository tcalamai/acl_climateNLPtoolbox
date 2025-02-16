from transformers import pipeline, AutoConfig
import os
from dotenv import load_dotenv

load_dotenv()

DATASETS = [
    'ClimaINS_ours',
    'ClimaINS',
    'climaQA',
    'climateBUG_data',
    'climateEng', 
    'climateFEVER_claim',
    'climateFEVER_evidence_climabench',
    'climateFEVER_evidence',
    'climateStance',
    'climatext_10k',
    'climatext_claim',
    'climatext', 
    'climatext_wiki',
    'climate_commitments_actions',
    'climate_detection',
    'climate_sentiment',
    'climate_specificity',
    'climate_tcfd_recommendations',
    'ClimaTOPIC',
    'contrarian_claims',
    'environmental_claims',
    'esgbert_action500',
    'esgbert_category_biodiversity',
    'esgbert_category_forest',
    'esgbert_category_nature',
    'esgbert_category_water',
    'esgbert_e', 
    'esgbert_g',
    'esgbert_s',
    'green_claims_3',
    'green_claims',
    'gw_stance_detection',
    'lobbymap_pages', 
    'lobbymap_query',
    'lobbymap_stance', 
    'logicClimate',
    'netzero_reduction',
    'sciDCC',
    'sustainable_signals_review'
]

HF_REPO = "anonymous"

def load_model(dataset_name: str, device=0, multilabel=False):
    """
    Load a finetuned model given its name.

    Args:
        model_name (str): The name of the model to load.
        device (int): The device to use for prediction (default: 0).

    Returns:
        A pipeline object that can be used for prediction.

    Raises:
        ValueError: If the model name is not found in the predefined DATASETS.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown model '{dataset_name}'. Available keys: {list(DATASETS)}")

    model_path = f"{HF_REPO}/{dataset_name}_42_distilRoBERTa"
    if multilabel:
        pipe = pipeline("text-classification", model=model_path, device=device, token=os.environ["HUGGINGFACE_TOKEN"], top_k=None)
        
        config = AutoConfig.from_pretrained(model_path, token=os.environ.get("HUGGINGFACE_TOKEN", None))
        if getattr(config, "problem_type", None) != "multi_label_classification":
            print("⚠ Warning: Model might not be set up for multi-label classification.")
    else:
        pipe = pipeline("text-classification", model=model_path, device=device, token=os.environ["HUGGINGFACE_TOKEN"])
 
    return pipe

def predict_text(text: str, dataset_name: str, device=0):
    """
    Predict the label for a given text using a given model.

    Args:
        text (str): The text to be classified
        model_name (str): The name of the model to use for prediction
        device (int): The device to use for prediction (default: 0)

    Returns:
        The predicted label
    """
    pipe = load_model(dataset_name, device=device)
    return pipe(text, truncation=True, 
        padding=True,
        max_length=512)[0]

def predict_batch(texts: list[str], dataset_name: str, device=0):
    """
    Predict the labels for a batch of texts using a given model.

    Args:
        texts (list[str]): The texts to be classified
        model_name (str): The name of the model to use for prediction
        device (int): The device to use for prediction (default: 0)

    Returns:
        A list of predicted labels
    """
    
    pipe = load_model(dataset_name, device=device)
    predictions =  pipe(texts, truncation=True, 
        padding=True,
        max_length=512)
    predictions = [pred['label'] for pred in predictions]
    return predictions

def predict_pair(text: list[str], query: list[str], dataset_name: str, device=0):    
    pipe = load_model(dataset_name, device=device)
    predictions =  pipe([{"text": text, "text_pair": query}], truncation=True, 
        padding=True,
        max_length=512)
    
    predictions = [pred['label'] for pred in predictions]
    return predictions

def predict_pair_batch(texts: list[str], queries: list[str], dataset_name: str, device=0):
    """
    Predict the labels for a batch of text-query pairs using a given model.

    Args:
        texts (list[str]): The texts to be classified
        queries (list[str]): The queries to be paired with texts
        dataset_name (str): The name of the model to use for prediction
        device (int): The device to use for prediction (default: 0)

    Returns:
        A list of predicted labels
    """
    pipe = load_model(dataset_name, device=device)
    predictions = pipe(
        [{"text": text, "text_pair": query} for text, query in zip(texts, queries)], 
        truncation=True, 
        padding=True,
        max_length=512
    )

    predictions = [pred['label'] for pred in predictions]
    return predictions

def predict_multilabel_batch(texts: list[str], dataset_name: str, device=0, threshold=0.5):
    """
    Classifies a batch of texts using a multi-label classification model.
    
    Args:
        texts (List[str]): List of input texts to classify.
        threshold (float): Probability threshold for selecting labels.

    Returns:
        List[dict]: A list of dictionaries, each containing labels and their scores for a given text.
    """
    pipe = load_model(dataset_name, device=device, multilabel=True)

    results = pipe(texts, truncation=True, 
        padding=True,
        max_length=512)
    filtered_results = []
    
    for text, predictions in zip(texts, results):
        filtered_labels = [label_data["label"]
                           for label_data in predictions if label_data["score"] > threshold]
        filtered_results.append(filtered_labels)
    
    return filtered_results


def predict_multilabel(text: str, dataset_name: str, device=0, threshold=0.5):
    return predict_multilabel_batch([text], dataset_name=dataset_name, device=device, threshold=threshold)
