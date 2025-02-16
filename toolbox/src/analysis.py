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
    import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_df_dashboard_story(df):
    # Set context for a compact dashboard
    sns.set_context("paper", font_scale=0.7)
    sns.set_style("whitegrid")
    
    # Filter climate-related and non-climate-related texts
    climate_df = df[df["climate"] == True]
    non_climate_df = df[df["climate"] == False]
    
    # Helper: normalize series
    def normalize_counts(series):
        return series / series.sum() if series.sum() != 0 else series

    # Create a 4x2 grid (8 subplots) for the dashboard
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    axes = axes.flatten()
    
    # Plot 1: Content Count Comparison (How much they talk about climate)
    content_counts = pd.Series({
        "Climate": len(climate_df),
        "Non-Climate": len(non_climate_df)
    })
    sns.barplot(x=content_counts.index, y=content_counts.values, ax=axes[0],
                palette=sns.color_palette("Spectral", n_colors=2)[::-1])
    axes[0].set_title("Content Count Comparison", fontsize=8)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count", fontsize=8)
    axes[0].tick_params(axis='x', labelsize=7)
    
    # Plot 2: ClimaTOPIC Distribution (What topics are discussed)
    # Map long category names to shorter ones.
    mapping = {"Governance and Data Management": "Gov & Data"}
    # Apply mapping to ClimaTOPIC values
    climatopic_short = climate_df["ClimaTOPIC"].apply(lambda x: mapping.get(x, x))
    climatopic_counts = normalize_counts(climatopic_short.value_counts())
    sns.barplot(x=climatopic_counts.index, y=climatopic_counts.values, 
                ax=axes[2], palette="Spectral")
    axes[2].set_title("ClimaTOPIC Distribution\n(Climate Only)", fontsize=8)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Prop.", fontsize=8)
    axes[2].tick_params(axis='x', labelrotation=45, labelsize=7)
    
    # Plot 3: Sentiment Distribution (Risk vs Opportunity framing)
    climate_sentiment = normalize_counts(climate_df["climate_sentiment"].value_counts())
    non_climate_sentiment = normalize_counts(non_climate_df["climate_sentiment"].value_counts())
    labels = climate_sentiment.index.union(non_climate_sentiment.index)
    data_sentiment = pd.DataFrame({
        "Climate": [climate_sentiment.get(label, 0) for label in labels],
        "Non-Climate": [non_climate_sentiment.get(label, 0) for label in labels]
    }, index=labels)
    data_sentiment.plot(kind="bar", ax=axes[1], 
                        color=sns.color_palette("Spectral", n_colors=2)[::-1],
                        width=0.8, legend=True)
    axes[1].set_title("Sentiment Distribution", fontsize=8)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Prop.", fontsize=8)
    axes[1].tick_params(axis='x', labelrotation=45, labelsize=7)
    
    # Plot 4: Environmental Claims Count (Are claims made?)
    claims_counts = pd.Series({
        "Env-claim": climate_df["Environmental claims"].sum(),
        "General": non_climate_df["Environmental claims"].sum()
    })
    sns.barplot(x=claims_counts.index, y=claims_counts.values, ax=axes[6],
                palette=sns.color_palette("Spectral", n_colors=2)[::-1])
    axes[6].set_title("Environmental Claims", fontsize=8)
    axes[6].set_xlabel("")
    axes[6].set_ylabel("Count", fontsize=8)
    axes[6].tick_params(axis='x', labelsize=7)
    
    # Plot 5: ESG Breakdown (Which ESG elements are emphasized)
    # Define a mapping for ESG pillars to the corresponding column names and the positive value.   
    climate_esg_df = climate_df[
            (climate_df['esgbert_e'] == "Environment") | 
            (climate_df['esgbert_s'] == "Social") |
            (climate_df['esgbert_g'] == "Governance")
            ].copy()
    esg_df = df[
            (df['esgbert_e'] == "Environment") | 
            (df['esgbert_s'] == "Social") |
            (df['esgbert_g'] == "Governance")
            ].copy()
    
    # Calculate percentages for each ESG category in the climate ESG texts
    total_climate = len(climate_esg_df)
    climate_E = (climate_esg_df['esgbert_e'] == "Environment").sum() / total_climate * 100 if total_climate else 0
    climate_S = (climate_esg_df['esgbert_s'] == "Social").sum() / total_climate * 100 if total_climate else 0
    climate_G = (climate_esg_df['esgbert_g'] == "Governance").sum() / total_climate * 100 if total_climate else 0

    # Calculate percentages for each ESG category in the overall ESG texts
    total_esg = len(esg_df)
    esg_E = (esg_df['esgbert_e'] == "Environment").sum() / total_esg * 100 if total_esg else 0
    esg_S = (esg_df['esgbert_s'] == "Social").sum() / total_esg * 100 if total_esg else 0
    esg_G = (esg_df['esgbert_g'] == "Governance").sum() / total_esg * 100 if total_esg else 0

    # Create a DataFrame with the results
    data = pd.DataFrame({
        "Climate ESG": [climate_E, climate_S, climate_G],
        "Overall ESG": [esg_E, esg_S, esg_G]
    }, index=["Environment", "Social", "Governance"])
    data = data.reset_index().rename(columns={"index": "ESG Category"})

    # Melt the DataFrame into a long format for seaborn
    data_melted = data.melt(id_vars="ESG Category", var_name="Group", value_name="Percentage")

    # Create a simple grouped bar plot on axes[4]
    sns.barplot(x="ESG Category", y="Percentage", hue="Group", data=data_melted, 
                palette=sns.color_palette("Spectral", 2)[::-1], ax=axes[4])
    axes[4].set_title("ESG Percentages", fontsize=8)
    axes[4].set_xlabel("")
    axes[4].set_ylabel("Percentage", fontsize=8)
    axes[4].tick_params(axis='x', labelsize=7)
        
    # Plot 6: TCFD Recommendations (Risk-focused recommendations)
    tcfd_counts = normalize_counts(climate_df["climate_tcfd_recommendations"].value_counts())
    sns.barplot(x=tcfd_counts.index, y=tcfd_counts.values, ax=axes[3],
                palette="Spectral")
    axes[3].set_title("TCFD Recommendations\n(Climate Only)", fontsize=8)
    axes[3].set_xlabel("")
    axes[3].set_ylabel("Prop.", fontsize=8)
    axes[3].tick_params(axis='x', labelrotation=45, labelsize=7)
    
    # Plot 7: Nature Mentions (Discussion of specific natural elements)
    forest = normalize_counts(climate_df["esgbert_category_forest"].value_counts())
    biodiv = normalize_counts(climate_df["esgbert_category_biodiversity"].value_counts())
    water = normalize_counts(climate_df["esgbert_category_water"].value_counts())
    data_nature = pd.DataFrame({
        "Forest": [forest.get("Forest", 0)],
        "Biodiversity": [biodiv.get("Biodiversity", 0)],
        "Water": [water.get("Water", 0)]
    })
    sns.barplot(data=data_nature, ax=axes[5], 
                palette=sns.color_palette("Spectral", n_colors=3)[::-1])
    axes[5].set_title("Nature Mentions", fontsize=8)
    axes[5].set_xlabel("")
    axes[5].set_ylabel("Prop.", fontsize=8)
    
    # Plot 8: Specificity by Claim (Are commitments specific or not?)
    # Separate the data based on Environmental claims.
    df_env_claim = df[df["Environmental claims"] == True].copy()
    df_not_env_claim = df[df["Environmental claims"] == False].copy()

    # Compute proportions for "Env Claims"
    total_env = len(df_env_claim)
    env_commit_prop = (df_env_claim["climate_commitments_actions"] == "commitment/action").sum() / total_env if total_env else 0
    env_specific_prop = (((df_env_claim["climate_commitments_actions"] == "commitment/action") & 
                           (df_env_claim["climate_specificity"] == "specific")).sum() / total_env) if total_env else 0

    # Compute proportions for "Non-Env Claims"
    total_non_env = len(df_not_env_claim)
    non_env_commit_prop = (df_not_env_claim["climate_commitments_actions"] == "commitment/action").sum() / total_non_env if total_non_env else 0
    non_env_specific_prop = (((df_not_env_claim["climate_commitments_actions"] == "commitment/action") & 
                               (df_not_env_claim["climate_specificity"] == "specific")).sum() / total_non_env) if total_non_env else 0

    # Build a DataFrame for plotting
    data_spec = pd.DataFrame({
        "Env Claims": [env_commit_prop, env_specific_prop],
        "Non-Env Claims": [non_env_commit_prop, non_env_specific_prop]
    }, index=["Commitment/Action", "Specific Commitment"]).reset_index().rename(columns={"index": "Measure"})

    # Melt the DataFrame to long format for seaborn
    data_spec_melted = data_spec.melt(id_vars="Measure", var_name="Claim Type", value_name="Proportion")

    # Create the grouped bar plot on axes[7] (or your desired subplot)
    sns.barplot(x="Claim Type", y="Proportion", hue="Measure", data=data_spec_melted, 
                palette="Spectral", ax=axes[7])
    axes[7].set_title("Specificity by Claim", fontsize=8)
    axes[7].set_xlabel("")
    axes[7].set_ylabel("Proportion", fontsize=8)
    axes[7].tick_params(axis='x', labelsize=7)
    axes[7].legend(title="Measure", fontsize=6, title_fontsize=7)

    
    plt.tight_layout()
    plt.show()
