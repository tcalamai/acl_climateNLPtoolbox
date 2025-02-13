import os
import joblib
import pandas as pd

MODEL_PATHS = {
    "ClimaINS_ours" : "models/tfidf/ClimaINS_ours_tfidf_+_LogReg.joblib",
    "ClimaINS" : "models/tfidf/ClimaINS_tfidf_+_LogReg.joblib" ,
    "climaQA" : "models/tfidf/climaQA_tfidf_+_LogReg.joblib" ,
    "climateBUG_data" : "models/tfidf/climateBUG_data_tfidf_+_LogReg.joblib" ,
    "climateEng" : "models/tfidf/climateEng_tfidf_+_LogReg.joblib" ,
    "climateFEVER_claim" : "models/tfidf/climateFEVER_claim_tfidf_+_LogReg.joblib" ,
    "climateFEVER_evidence_climabench" : "models/tfidf/climateFEVER_evidence_climabench_tfidf_+_LogReg.joblib" ,
    "climateFEVER_evidence" : "models/tfidf/climateFEVER_evidence_tfidf_+_LogReg.joblib" ,
    "climateStance" : "models/tfidf/climateStance_tfidf_+_LogReg.joblib" ,
    "climatext_10k" : "models/tfidf/climatext_10k_tfidf_+_LogReg.joblib" ,
    "climatext_claim" : "models/tfidf/climatext_claim_tfidf_+_LogReg.joblib" ,
    "climatext" : "models/tfidf/climatext_tfidf_+_LogReg.joblib" ,
    "climatext_wiki" : "models/tfidf/climatext_wiki_tfidf_+_LogReg.joblib" ,
    "climate_commitments_actions" : "models/tfidf/climate_commitments_actions_tfidf_+_LogReg.joblib" ,
    "climate_detection" : "models/tfidf/climate_detection_tfidf_+_LogReg.joblib" ,
    "climate_sentiment" : "models/tfidf/climate_sentiment_tfidf_+_LogReg.joblib" ,
    "climate_specificity" : "models/tfidf/climate_specificity_tfidf_+_LogReg.joblib" ,
    "climate_tcfd_recommendations" : "models/tfidf/climate_tcfd_recommendations_tfidf_+_LogReg.joblib" ,
    "ClimaTOPIC" : "models/tfidf/ClimaTOPIC_tfidf_+_LogReg.joblib" ,
    "contrarian_claims" : "models/tfidf/contrarian_claims_tfidf_+_LogReg.joblib" ,
    "environmental_claims" : "models/tfidf/environmental_claims_tfidf_+_LogReg.joblib" ,
    "esgbert_action500" : "models/tfidf/esgbert_action500_tfidf_+_LogReg.joblib" ,
    "esgbert_category_biodiversity" : "models/tfidf/esgbert_category_biodiversity_tfidf_+_LogReg.joblib" ,
    "esgbert_category_forest" : "models/tfidf/esgbert_category_forest_tfidf_+_LogReg.joblib" ,
    "esgbert_category_nature" : "models/tfidf/esgbert_category_nature_tfidf_+_LogReg.joblib" ,
    "esgbert_category_water" : "models/tfidf/esgbert_category_water_tfidf_+_LogReg.joblib" ,
    "esgbert_e" : "models/tfidf/esgbert_e_tfidf_+_LogReg.joblib" ,
    "esgbert_g" : "models/tfidf/esgbert_g_tfidf_+_LogReg.joblib" ,
    "esgbert_s" : "models/tfidf/esgbert_s_tfidf_+_LogReg.joblib" ,
    "green_claims_3" : "models/tfidf/green_claims_3_tfidf_+_LogReg.joblib" ,
    "green_claims" : "models/tfidf/green_claims_tfidf_+_LogReg.joblib" ,
    "gw_stance_detection" : "models/tfidf/gw_stance_detection_tfidf_+_LogReg.joblib" ,
    "lobbymap_pages" : "models/tfidf/lobbymap_pages_tfidf_+_LogReg.joblib" ,
    "lobbymap_stance" : "models/tfidf/lobbymap_stance_tfidf_+_LogReg.joblib" ,
    "logicClimate" : "models/tfidf/logicClimate_tfidf_+_LogReg.joblib" ,
    "netzero_reduction" : "models/tfidf/netzero_reduction_tfidf_+_LogReg.joblib" ,
    "sciDCC" : "models/tfidf/sciDCC_tfidf_+_LogReg.joblib" ,
    "sustainable_signals_review" : "models/tfidf/sustainable_signals_review_tfidf_+_LogReg.joblib",
    "lobbymap_query" : "models/tfidf/lobbymap_query_tfidf_+_LogReg.joblib" #TODO: add the model
}

MAPPING_MULTILABEL = {
    "lobbymap_query": {'alignment_with_ipcc_on_climate_action': 0, 'carbon_tax': 1, 'communication_of_climate_science': 2, 'disclosure_on_relationships': 3, 'emissions_trading': 4, 'energy_and_resource_efficiency': 5, 'energy_transition_&_zero_carbon_technologies': 6, 'ghg_emission_regulation': 7, 'land_use': 8, 'renewable_energy': 9, 'support_of_un_climate_process': 10, 'supporting_the_need_for_regulations': 11, 'transparency_on_legislation': 12},
    "logicClimate": {'ad hominem': 0, 'ad populum': 1, 'appeal to emotion': 2, 'circular reasoning': 3, 'equivocation': 4, 'fallacy of credibility': 5, 'fallacy of extension': 6, 'fallacy of logic': 7, 'fallacy of relevance': 8, 'false causality': 9, 'false dilemma': 10, 'faulty generalization': 11, 'intentional': 12}
}


def load_model(model_name: str):
    """
    Load a machine learning model pipeline from disk.

    Args:
        model_name: The name of the model to load.

    Returns:
        The loaded model pipeline.

    Raises:
        ValueError: If the model name is not found in the predefined MODEL_PATHS.
    """

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model '{model_name}'. Available keys: {list(MODEL_PATHS.keys())}")

    model_path = MODEL_PATHS[model_name]

    # Build absolute path if necessary
    abs_path = os.path.join(os.path.dirname(__file__), model_path)
    loaded_pipeline = joblib.load(abs_path)
    return loaded_pipeline


def predict_text(text: str, dataset_name: str):
    """
    Predict the label for a given text using a given model.

    Args:
        text: The text to be classified
        model_name: The name of the model to use for prediction

    Returns:
        The predicted label
    """
    model = load_model(dataset_name)
    return model.predict([text])[0]


def predict_batch(texts: list[str], dataset_name: str):
    """
    Predict the labels for a batch of texts using a given model.

    Args:
        text: A list of texts to be classified
        model_name: The name of the model to use for prediction

    Returns:
        A list of predicted labels for each text
    """
    model = load_model(dataset_name)
    return model.predict(texts)

def predict_multilabel_batch(texts: list[str], dataset_name: str):
    pred = predict_batch(texts, dataset_name=dataset_name)

    reversed_mapping = {v: k for k, v in MAPPING_MULTILABEL[dataset_name].items()}
    
    decoded_labels = []
    
    for row in pred:
        labels = [reversed_mapping[idx] for idx, value in enumerate(row) if value == 1]
        decoded_labels.append(labels)
    
    return decoded_labels


def predict_text_multi(text: str, dataset_name: str):
    return predict_multilabel_batch([text], dataset_name=dataset_name)

def predict_pair(text: list[str], query: list[str], dataset_name: str, device=0):    
    df_input = pd.DataFrame({"text": [text], "query": [query]})
    
    model = load_model(dataset_name)
    return model.predict(df_input)

def predict_pair_batch(texts: list[str], queries: list[str], dataset_name: str):
    """
    Predict the labels for a batch of text-query pairs using a given model.

    Args:
        texts (list[str]): The texts to be classified
        queries (list[str]): The queries to be paired with texts
        dataset_name (str): The name of the model to use for prediction

    Returns:
        A list of predicted labels
    """
    df_input = pd.DataFrame({"text": texts, "query": queries})

    model = load_model(dataset_name)
    predictions = model.predict(df_input)
    return predictions
