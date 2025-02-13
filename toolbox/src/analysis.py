import os
from src.pdf_parser import processed_files_iterator
from src.tasks import *
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_folder(model_name="tfidf", path="case_study/data/", save_path="case_study/machine_readable_data", report_path="case_study/analysis/"):
    if model_name not in ["tfidf", "distilRoBERTa", "LLM"]:
        raise ValueError(f"Unknown model name '{model_name}'. Available model names: ['tfidf', 'distilRoBERTa', 'LLM']")
    
    for filename, df in processed_files_iterator(save_path=save_path):
        print(f"Processing {filename}")
        
        texts = list(df['text'].values)
        
        climate_datasets = [
            "climatext_10k", 
            # "climatext_claim", 
            "climatext_wiki", 
            # "climatext", 
            "climateBUG_data", 
            "climate_detection",
            # "sustainable_signals_review"
        ]
        
        df['climate'] = climate_related(
            texts=texts,
            datasets=climate_datasets,
            model_name=model_name,
            mode="any"
        )
        
        topic_list = ["climate_tcfd_recommendations",
                "ClimaTOPIC",
                "esgbert_category_biodiversity",
                "esgbert_category_forest",
                # "esgbert_category_nature", # contained in other topics
                "esgbert_category_water",
                "esgbert_e",
                "esgbert_s",
                "esgbert_g",
                # "sciDCC", # for news
                # "climateEng" # for tweets
            ]
        
        topic_labels = topic_detection(
            texts=texts,
            datasets=topic_list,
            model_name=model_name,
            mode="list"
        )
        

        topic_labels = np.array(topic_labels)  # Convert to NumPy array if it's not already
        for i, topic in enumerate(topic_list):
            df[topic] = topic_labels[:, i]  # Directly assign the column

        
        df['Environmental claims'] = green_claims_detection(
            texts=texts,
            datasets=["environmental_claims"], # green_claims is for tweets
            model_name=model_name,
            mode="majority"
        )
        
        topic_list = ["climate_sentiment", "climate_specificity", "climate_commitments_actions", "netzero_reduction"]
        topic_labels = characteristics_classification(
            texts=texts,
            datasets=topic_list,
            model_name=model_name,
            mode="list"
        )
        
        topic_labels = np.array(topic_labels)  # Convert to NumPy array if it's not already
        for i, topic in enumerate(topic_list):
            df[topic] = topic_labels[:, i]  # Directly assign the column

        df['stance'] = stance_classification(
            texts=texts,
            model_name=model_name,
            datasets=["climateStance", "gw_stance_detection"],
            mode="any"
        )
        
        doc_pages = df.groupby(by=['page_number', "filename"])['text'].apply(lambda x: "\n".join(x)).reset_index()
        
        texts = list(doc_pages['text'].values)
        
        doc_pages['policy_stance'] = policy_stance(
            texts=texts,
            model_name=model_name
        )
        
        # doc_pages.to_parquet(report_path + filename.replace(".parquet", f"_pages_{model_name}.parquet")) # Can not be saved because PyArrow does not allow [1, [1,2]] should be 2 columns or [[1], [1,2]]
        df.to_parquet(report_path + filename.replace(".parquet", f"_paragraphs_{model_name}.parquet"))
        

def visualize(model_name="tfidf", report_path="case_study/analysis/"):
    for filename in os.listdir(report_path):
        if filename.endswith(f"_paragraphs_{model_name}.parquet"):
            print(f"Visualizing {filename}")
            visualize_df(pd.read_parquet(report_path + filename))

def visualize_df(df):
    # Filter climate-related and non-climate-related texts
    climate_df = df[df["climate"] == True]
    non_climate_df = df[df["climate"] == False]
    
    # Normalize function
    def normalize_counts(series):
        return series / series.sum()
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    
    # ClimaTOPIC Distribution (Only for Climate-Related Texts)
    climatopic_counts = normalize_counts(climate_df["ClimaTOPIC"].value_counts())
    sns.barplot(x=climatopic_counts.index, y=climatopic_counts.values, ax=axes[0, 0], color="blue")
    axes[0, 0].set_title("Normalized ClimaTOPIC Distribution (Climate-Related Only)")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), ha="right", rotation=45)
    
    # Risk/Opportunity/Neutral Profile for Climate and Non-Climate Texts
    climate_sentiment_counts = normalize_counts(climate_df["climate_sentiment"].value_counts())
    non_climate_sentiment_counts = normalize_counts(non_climate_df["climate_sentiment"].value_counts())
    labels = climate_sentiment_counts.index.union(non_climate_sentiment_counts.index)
    data = pd.DataFrame({
        "Climate-Related": [climate_sentiment_counts.get(label, 0) for label in labels],
        "Non-Climate-Related": [non_climate_sentiment_counts.get(label, 0) for label in labels]
    }, index=labels)
    data.plot(kind="bar", ax=axes[0, 1], color=["green", "gray"], width=0.8)
    axes[0, 1].set_title("Normalized Sentiment Distribution")
    
    # ESG Breakdown for Climate vs Non-Climate
    esg_categories = ["esgbert_e", "esgbert_s", "esgbert_g"]
    esg_labels = ["Environmental (E)", "Social (S)", "Governance (G)"]
    climate_esg_counts = [normalize_counts(climate_df[cat].value_counts()).get("Environment", 0) for cat in esg_categories]
    non_climate_esg_counts = [normalize_counts(non_climate_df[cat].value_counts()).get("Environment", 0) for cat in esg_categories]
    df_esg = pd.DataFrame({
        "Climate-Related": climate_esg_counts,
        "Non-Climate-Related": non_climate_esg_counts
    }, index=esg_labels)
    df_esg.plot(kind="bar", ax=axes[1, 0], color=["green", "gray"], width=0.8)
    axes[1, 0].set_title("Normalized ESG Breakdown")
    
    # TCFD Recommendations (Only for Climate-Related Texts)
    tcfd_counts = normalize_counts(climate_df["climate_tcfd_recommendations"].value_counts())
    sns.barplot(x=tcfd_counts.index, y=tcfd_counts.values, ax=axes[1, 1], color="blue")
    axes[1, 1].set_title("Normalized TCFD Recommendation Distribution")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), ha="right", rotation=45)
    
    # Forest, Biodiversity, Water Proportion (Only for Climate-Related Texts)
    forest_counts = normalize_counts(climate_df["esgbert_category_forest"].value_counts())
    biodiversity_counts = normalize_counts(climate_df["esgbert_category_biodiversity"].value_counts())
    water_counts = normalize_counts(climate_df["esgbert_category_water"].value_counts())
    data_nature = pd.DataFrame({
        "Forest": [forest_counts.get("Forest", 0)],
        "Biodiversity": [biodiversity_counts.get("Biodiversity", 0)],
        "Water": [water_counts.get("Water", 0)]
    })
    sns.barplot(data=data_nature, ax=axes[2, 0], palette=["brown", "green", "blue"])
    axes[2, 0].set_title("Normalized Forest, Biodiversity, and Water Mentions")
    
    # Climate vs Non-Climate Count Comparison
    climate_counts = [len(climate_df), len(non_climate_df)]
    sns.barplot(x=["Climate-Related", "Non-Climate-Related"], y=climate_counts, ax=axes[2, 1], color="purple")
    axes[2, 1].set_title("Climate vs Non-Climate Content Count")
    
    plt.tight_layout()
    plt.show()
