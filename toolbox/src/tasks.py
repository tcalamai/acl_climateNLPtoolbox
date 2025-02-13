import joblib
from src import tfidf, finetuned

DATASET_LABELS = {
    # TODO: check the mapping for 0/1
    "climatext_10k": {
        "positive": "1", 
        "negative": "0"
    },
    "climatext_claim": {
        "positive": "1", 
        "negative": "0"
    },
    "climatext_wiki": {
        "positive": "1", 
        "negative": "0"
    },
    "climatext": {
        "positive": "1", 
        "negative": "0"
    },
    "climateBUG_data": {
        "positive": "climate", 
        "negative": "non-climate"
    },
    "climate_detection": {
        "positive": "yes", 
        "negative": "no"
    },
    "green_claims": {
        "positive": "green", 
        "negative": "not_green"
    },
    "environmental_claims": {
        "positive": "1", 
        "negative": "0"
    },
    "sustainable_signals_review":{
        "positive": "1",
        "negative": "0"
    },
    "climateStance":{
        "Neutral/Ambiguous": "Neutral/Ambiguous",
        "Support": "Support",
        "Denies": "Denies"
    },
    "gw_stance_detection":{
        "Neutral/Ambiguous": "neutral",
        "Support": "agrees",
        "Denies": "disagrees"
    }
}

TOPIC_LABEL_MAP = {
    "esgbert_category_biodiversity": {
        0: "Not-biodiversity",
        1: "Biodiversity"
    },
    "esgbert_category_forest": {
        0: "Not-forest",
        1: "Forest"
    },
    "esgbert_category_nature": {
        0: "Not-nature",
        1: "Nature"
    },
    "esgbert_category_water": {
        0: "Not-water",
        1: "Water"
    },
    "esgbert_e": {
        0: "Not-environment",
        1: "Environment"
    },
    "esgbert_s": {
        0: "Not-social",
        1: "Social"
    },
    "esgbert_g": {
        0: "Not-governance",
        1: "Governance"
    },
    "climateEng": {
        0: "General",
        1: "Politics",
        2: "Ocean/Water",
        3: "Agriculture/Forestry",
        4: "Disaster"
    },
    "climate_commitments_actions": {
        "no": "not-commitment/action",
        "yes": "commitment/action"
    },
    "esgbert_action500": {
        0: "Not-action",
        1: "Action"
    },
    "contrarian_claims": {
      "0_0": "No claim, No claim",

      "1_1":"Global warming is not happening, Ice/permafrost/snow cover isn’t melting",
      "1_2":"Global warming is not happening, We’re heading into an ice age/global cooling",
      "1_3":"Global warming is not happening, Weather is cold/snowing",
      "1_4":"Global warming is not happening, Climate hasn’t warmed/changed over the last (few) decade(s)",
      "1_6":"Global warming is not happening, Sea level rise is exaggerated/not accelerating",
      "1_7":"Global warming is not happening, Extreme weather isn’t increasing/has happened before/isn’t linked to climate change",

      "2_1":"Human greenhouse gases are not causing climate change, It’s natural cycles/variation",
      "2_3":"Human greenhouse gases are not causing climate change, There’s no evidence for greenhouse effect/carbon dioxide driving climate change",

      "3_1":"Climate impacts/global warming is beneficial/not bad, Climate sensitivity is low/negative feedbacks reduce warming",
      "3_2":"Climate impacts/global warming is beneficial/not bad, Species/plants/reefs aren’t showing climate impacts/are benefiting from climate change",
      "3_3":"Climate impacts/global warming is beneficial/not bad, CO2 is beneficial/not a pollutant",

      "4_1":"Climate solutions won’t work, Climate policies (mitigation or adaptation) are harmful",
      "4_2":"Climate solutions won’t work, Climate policies are ineffective/flawed",
      "4_4":"Climate solutions won’t work, Clean energy technology/biofuels won’t work",
      "4_5":"Climate solutions won’t work, People need energy (e.g. from fossil fuels/nuclear)",

      "5_1":"Climate movement/science is unreliable, Climate-related science is unreliable/uncertain/unsound (data, methods & models)",
      "5_2":"Climate movement/science is unreliable, Climate movement is unreliable/alarmist/corrupt"
    },
    "climateStance": {
        0: "Neutral/Ambiguous",
        1: "Support",
        2: "Denies"
    },
}

def model_selector(model, mode="classification"):
    """
    Select a model given its name.

    Args:
        model (str): The name of the model to use for prediction

    Returns:
        function: The selected model

    Raises:
        ValueError: If the model name is not found in the predefined MODEL_PATHS
    """
    if mode == "classification":
        if model == "tfidf":
                return tfidf.predict_batch
        elif model == "distilRoBERTa":
                return finetuned.predict_batch
        else:
            raise ValueError(f"Unknown model '{model}'. Available models: {['tfidf', 'distilRoBERTa', 'LLM']}")
    elif mode == "pairs":
        if model == "tfidf":
                return tfidf.predict_pair_batch
        elif model == "distilRoBERTa":
                return finetuned.predict_pair_batch
        else:
            raise ValueError(f"Unknown model '{model}'. Available models: {['tfidf', 'distilRoBERTa', 'LLM']}")
    elif mode == "multilabel":
        if model == "tfidf":
                return tfidf.predict_multilabel_batch
        elif model == "distilRoBERTa":
                return finetuned.predict_multilabel_batch
        else:
            raise ValueError(f"Unknown model '{model}'. Available models: {['tfidf', 'distilRoBERTa', 'LLM']}")
    else:
        raise ValueError(f"Unknown model '{mode}'. Available models: {['classification', 'paris']}")


def voting(texts: list[str], datasets: list[str], allowed_datasets: list[str], model_name="tfidf", mode="majority", boolean=True):
    if isinstance(texts, str):
        texts = [texts]

    if isinstance(datasets, str):
        datasets = [datasets]

    for dataset in datasets:
        if dataset not in allowed_datasets:
            raise ValueError(
                f"Dataset '{dataset}' is not in the list of allowed datasets: {allowed_datasets}"
            )
    
    all_predictions_by_dataset = []
    
    for dataset_name in datasets:
        label_map = DATASET_LABELS.get(dataset_name)
        if not label_map:
            raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_LABELS.")
        
        predicted_labels = model_selector(model_name)(texts, dataset_name)
        
        if boolean:
            dataset_boolean_preds = [
                (str(pred_label) == label_map["positive"])
                for pred_label in predicted_labels
            ]
            all_predictions_by_dataset.append(dataset_boolean_preds)
        else:
            all_predictions_by_dataset.append(predicted_labels)
    
    num_texts = len(texts)
    final_results = []

    for i in range(num_texts):
        preds_for_this_text = [pred_list[i] for pred_list in all_predictions_by_dataset]
        
        if mode == "majority":
            count_true = sum(preds_for_this_text)
            count_false = len(preds_for_this_text) - count_true
            final_results.append(count_true > count_false)
        elif mode == "any":
            final_results.append(any(preds_for_this_text))
        elif mode == "list":
            final_results.append(preds_for_this_text)
        else:
            raise ValueError(
                "Invalid mode. Please choose one of: 'majority', 'any', or 'list'."
            )

    return final_results

def classification(texts: str, datasets=list[str], allowed_datasets=list[str], model_name=str, mode=str):
    if isinstance(texts, str):
        texts = [texts]
        
    if isinstance(datasets, str):
        datasets = [datasets]
        
    for dataset in datasets:
        if dataset not in allowed_datasets:
            raise ValueError(f"Dataset '{dataset}' is not in the list of allowed datasets: {allowed_datasets}")
    
    all_predictions_by_dataset = []
    
    for dataset_name in datasets:
        predicted_labels = model_selector(model_name)(texts, dataset_name)
        
        if dataset_name in TOPIC_LABEL_MAP.keys():
            predicted_labels = [TOPIC_LABEL_MAP[dataset_name][pred] for pred in predicted_labels]
        
        all_predictions_by_dataset.append(predicted_labels)
    
    # 3. Combine predictions across datasets for each text, according to `mode`
    num_texts = len(texts)
    final_results = []

    for i in range(num_texts):
        # Gather the i-th prediction from each dataset
        preds_for_this_text = [pred_list[i] for pred_list in all_predictions_by_dataset]
        
        if mode == "list":
            # Return the entire list of booleans for this text
            final_results.append(preds_for_this_text)
        else:
            raise ValueError(
                "Invalid mode. Please choose one of: 'majority', 'any', or 'list'."
            )

    return final_results


def climate_related(
    texts: list[str],
    datasets=["climatext_10k", "climatext_claim", "climatext_wiki", "climatext", 
              "climateBUG_data", "climate_detection", "sustainable_signals_review"],
    model_name="tfidf",
    mode="majority"
):
    allowed_datasets = [
        "climatext_10k", 
        "climatext_claim", 
        "climatext_wiki", 
        "climatext", 
        "climateBUG_data", 
        "climate_detection",
        "sustainable_signals_review"
    ]
    return voting(
        texts=texts,
        datasets=datasets,
        allowed_datasets=allowed_datasets,
        model_name=model_name,
        mode=mode,
        boolean=True
    )
  


def topic_detection(
    texts: str,
    datasets=["climate_tcfd_recommendations",
                "ClimaTOPIC",
                "esgbert_category_biodiversity",
                "esgbert_category_forest",
                "esgbert_category_nature",
                "esgbert_category_water",
                "esgbert_e",
                "esgbert_s",
                "esgbert_g",
                "sciDCC",
                "climateEng",
                        ],
    model_name="tfidf",
    mode="list"
):
    allowed_datasets = ["climate_tcfd_recommendations",
                            "ClimaTOPIC",
                            "esgbert_category_biodiversity",
                            "esgbert_category_forest",
                            "esgbert_category_nature",
                            "esgbert_category_water",
                            "esgbert_e",
                            "esgbert_s",
                            "esgbert_g",
                            "sciDCC",
                            "climateEng",
                            ]
    return classification(
            texts=texts,
            datasets=datasets,
            allowed_datasets=allowed_datasets,
            model_name=model_name,
            mode=mode,
        )


def green_claims_detection(
    texts: str,
    datasets=["green_claims", "environmental_claims"],
    model_name="tfidf",
    mode="majority"
):
    allowed_datasets = ["green_claims", "environmental_claims"]
    return voting(
        texts=texts,
        datasets=datasets,
        allowed_datasets=allowed_datasets,
        model_name=model_name,
        mode=mode,
        boolean=True
    )


def characteristics_classification(
    texts: str,
    datasets=["green_claims_3",
            "climate_sentiment",
            "climate_specificity",
            "climate_commitments_actions",
            "netzero_reduction"
            ],
    model_name="tfidf",
    mode="list"
):
    allowed_datasets = ["green_claims_3",
                        "climate_sentiment",
                        "climate_specificity",
                        "climate_commitments_actions",
                        "netzero_reduction",
                        "esgbert_action500"
                        ]
    return classification(
            texts=texts,
            datasets=datasets,
            allowed_datasets=allowed_datasets,
            model_name=model_name,
            mode=mode,
        )



def stance_classification(
    texts: str,
    datasets=["climateStance", "gw_stance_detection"],
    model_name="tfidf",
    mode="majority"
):
    allowed_datasets = ["climateStance", "gw_stance_detection"]
    return voting(
        texts=texts,
        datasets=datasets,
        allowed_datasets=allowed_datasets,
        model_name=model_name,
        mode=mode,
        boolean=False
    )
    
def single_dataset(texts: list[str], dataset_name: str, model_name=str):
    predicted_label = model_selector(model_name)(texts, dataset_name)
    
    if dataset_name in TOPIC_LABEL_MAP.keys():
        predicted_label = [TOPIC_LABEL_MAP[dataset_name][pred] for pred in predicted_label]
        
    return predicted_label
        
def contrarian_claims(
    texts: str,
    model_name="tfidf",
):
    return single_dataset(texts, "contrarian_claims", model_name=model_name)

def logicClimate(
    texts: str,
    model_name="tfidf",
):
    return model_selector(model=model_name, mode="multilabel")(texts, dataset_name="logicClimate")

def policy_stance(
    texts: list[str],
    model_name="tfidf",
):
    """
    Process a batch of texts to predict policy stance efficiently.
    
    Args:
        texts (list[str]): A batch of texts to process.
        model_name (str): The name of the model to use.
    
    Returns:
        list: A list of predictions, where each entry contains
              [predicted_label, queries (if applicable), stance (if applicable)].
    """
    # Select the classification model
    classify_model = model_selector(model=model_name, mode="classification")
    
    # Predict labels for all texts in batch
    predicted_labels = classify_model(texts, dataset_name="lobbymap_pages")
    
    # Identify positive predictions
    positive_indices = [i for i, label in enumerate(predicted_labels) if label]
    positive_texts = [texts[i] for i in positive_indices]
    
    # Initialize outputs
    queries_list = [None] * len(texts)
    stance_list = [None] * len(texts)
    
    if positive_texts:
        # Get queries for positive texts in batch
        query_model = model_selector(model_name, mode="multilabel")
        queries_batch = query_model(texts=positive_texts, dataset_name="lobbymap_query")
        
        # Get stance predictions for each text-query pair in batch mode
        stance_model = model_selector(model_name, mode="pairs")
        
        stance_batch = [
            stance_model([positive_texts[i]] * len(queries), queries, dataset_name="lobbymap_stance")
            for i, queries in enumerate(queries_batch)
        ]
        
        # Assign results to corresponding indices
        for idx, original_idx in enumerate(positive_indices):
            queries_list[original_idx] = queries_batch[idx]
            stance_list[original_idx] = stance_batch[idx]
    
    # Construct final predictions
    predictions = [
        [predicted_labels[i], queries_list[i], stance_list[i]] for i in range(len(texts))
    ]
    
    return predictions

