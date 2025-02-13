import os.path

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

from transformers import AutoTokenizer
from cleantext import clean


from imblearn.under_sampling import RandomUnderSampler



def clean_text(text):
    cleaned_text = clean(text,
                         fix_unicode=True,  # fix various unicode errors
                         to_ascii=True,  # transliterate to closest ASCII representation
                         lower=False,  # lowercase text
                         no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
                         no_urls=True,  # replace all URLs with a special token
                         no_emails=True,  # replace all email addresses with a special token
                         no_phone_numbers=True,  # replace all phone numbers with a special token
                         no_numbers=False,  # replace all numbers with a special token
                         no_digits=False,  # replace all digits with a special token
                         no_currency_symbols=False,  # replace all currency symbols with a special token
                         no_punct=False,  # remove punctuations
                         # replace_with_punct="",
                         replace_with_url="<URL>",
                         replace_with_email="<EMAIL>",
                         replace_with_phone_number="<PHONE>",
                         # replace_with_number="<NUMBER>",
                         # replace_with_digit="0",
                         # replace_with_currency_symbol="<CUR>",
                         lang="en"  # set to 'de' for German special handling
                         )
    return cleaned_text


def split_stratify_time(df, test_size, dev_size):
    # split into labels
    df_train = []
    df_test = []
    df_dev = []

    df = df.sort_values(by="Date")

    for label in df['label'].unique():
        df_label = df[df['label'] == label].copy()

        a = int(len(df_label) * (1 - test_size))
        c = int((len(df_label) - a) * dev_size)
        b = len(df_label) - a - c

        split_train_label = df_label.iloc[:a]
        split_dev_label = df_label.iloc[a:a + b]
        split_test_label = df_label.iloc[a + b:a + b + c]

        df_train += [split_train_label]
        df_test += [split_dev_label]
        df_dev += [split_test_label]

    train_split = pd.concat(df_train)
    test_split = pd.concat(df_test)
    dev_split = pd.concat(df_dev)

    return train_split, test_split, dev_split

def reconstruct_page(dataset_df):
    exploded_train = dataset_df[['document_id', 'sentences']].explode('sentences')

    exploded_train['page_idx'] = exploded_train['sentences'].apply(lambda x: x['page_idx'])
    exploded_train['sentence_id'] = exploded_train['sentences'].apply(lambda x: x['sentence_id'])
    exploded_train['block_idx'] = exploded_train['sentences'].apply(lambda x: x['block_idx'])
    exploded_train['text'] = exploded_train['sentences'].apply(lambda x: x['text'])

    page_inputs = exploded_train.groupby(by=['document_id', 'page_idx', 'block_idx'])[
        'text'].sum().reset_index()
    page_inputs = page_inputs.groupby(by=['document_id', 'page_idx'])['text'].apply(lambda x: "\\n".join(x))

    return page_inputs.reset_index()

def get_page_idx(l):
    list_of_lists = [e['page_indices'] for e in l]
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    return list(set(flattened_list))

def get_page_query_map(ds):
    ds_exploded = ds.explode('evidences')
    ds_exploded['page_indices'] = ds_exploded['evidences'].apply(lambda x: x['page_indices'])
    ds_exploded['query'] = ds_exploded['evidences'].apply(lambda x: x['query'])
    mapping = ds_exploded[['document_id', 'page_indices', 'query']].explode('page_indices')
    mapping = mapping.groupby(by=['document_id', 'page_indices'])['query'].apply(lambda x: list(x))
    return mapping.reset_index()

def get_page_stance_map(ds):
    ds_exploded = ds.explode('evidences')
    ds_exploded['page_indices'] = ds_exploded['evidences'].apply(lambda x: x['page_indices'])
    ds_exploded['query'] = ds_exploded['evidences'].apply(lambda x: x['query'])
    ds_exploded['stance'] = ds_exploded['evidences'].apply(lambda x: x['stance'])
    mapping = ds_exploded[['document_id', 'page_indices', 'query', 'stance']].explode('page_indices')
    return mapping.reset_index()

class DatasetBuilder():
    def __init__(self, seed=42):
        print('initializing builder')
        self.seed = seed
        self.train_test_split = 0.2
        self.test_dev_split = 0.5
        self.path = "data"

        self.datasets = {
            "netzero_reduction": self.netzero_reduction,
            "climate_specificity": self.climate_specificity,
            "climate_sentiment": self.climate_sentiment,
            "climate_commitments_actions": self.climate_commitments_actions,
            "climate_detection": self.climate_detection, #Climate classification
            "climate_tcfd_recommendations": self.climate_tcfd_recommendations,
            "climatext": self.climatext, #Climate classification
            "climatext_wiki": self.climatext_wiki,
            "climatext_claim": self.climatext_claim,
            "climatext_10k": self.climatext_10k,
            "environmental_claims": self.environmental_claims,
            "ClimaTOPIC": self.ClimaTOPIC,
            "climateFEVER_claim": self.climateFEVER_claim,
            "climateBUG_data": self.climateBUG_data,
            "lobbymap_pages": self.lobbymap_pages,
            "sustainable_signals_review": self.sustainable_signals_review, #Climate classification
            "esgbert_e":self.esgbert_e,
            "esgbert_s": self.esgbert_s,
            "esgbert_g": self.esgbert_g,
            "esgbert_action500": self.esgbert_action500,
            "esgbert_category_water": self.esgbert_category_water,
            "esgbert_category_forest": self.esgbert_category_forest,
            "esgbert_category_biodiversity": self.esgbert_category_biodiversity,
            "esgbert_category_nature": self.esgbert_category_nature,
            "sciDCC": self.sciDCC,
            "green_claims": self.green_claims,
            "green_claims_3": self.green_claims_3,
            "contrarian_claims": self.contrarian_claims,
            "climateStance": self.climateStance,
            "climateEng": self.climateEng,
            # "climateStance_Reddit": self.climateStance_Reddit, small, would be included in climatestance ? or as a transfer learning task ?
            # "climateEng_Reddit": self.climateEng_Reddit,
            "ClimaINS": self.climaInsurance,
            "ClimaINS_ours": self.climaInsuranceOurs,
            "gw_stance_detection": self.climateStance_gwsd,
        }

        self.multilabel_datasets = {
            "lobbymap_query": self.lobbymap_query, #(self.lobbymap_query_multilabel, self.get_lobbymap_queries),
            "logicClimate": self.logicClimate,
            # Should add the E,S,G datasets (as a multilabel dataset)
            # Should add the Water, Forest, Biodiversity datasets (as a multilabel dataset)
        }

        self.stance_datasets = {
            "lobbymap_stance": self.lobbymap_stance,
        }

        self.relation_datasets = {
            "climateFEVER_evidence": self.climateFEVER_evidence,
            "climateFEVER_evidence_climabench": self.climateFEVER_evidence_climabench,
            "climaQA": self.OurClimaQA,
        }

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


    def count_tokens(self, text):
        if not isinstance(text, str) or not text.strip():
            print(f"Invalid text encountered: {text}")
            return 0  # or handle as needed
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)

    def __iter__(self):
        self.dataset_names = iter(self.datasets.keys())
        return self

    def __next__(self):
        try:
            dataset_name = next(self.dataset_names)
        except StopIteration:
            raise StopIteration
        train, test, dev = self.datasets[dataset_name]()

        return (dataset_name, (train, test, dev))

    def netzero_reduction(self):
        raw_dataset = load_dataset("climatebert/netzero_reduction_data")['train']

        texts = [record['text'] for record in raw_dataset]
        labels = [record['target'] for record in raw_dataset]

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=self.train_test_split,
                                                            random_state=self.seed, stratify=labels, shuffle=True)
        X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=self.test_dev_split,
                                                        random_state=self.seed, stratify=y_test, shuffle=True)

        train = pd.DataFrame({"text": X_train, "label": y_train})
        test = pd.DataFrame({"text": X_test, "label": y_test})
        dev = pd.DataFrame({"text": X_dev, "label": y_dev})

        return train, test, dev

    def train_test_huggingface_datasets(self, raw_dataset, text_column='text', label_column="label"):
        train_dataset = raw_dataset['train']
        id2label = train_dataset.features[label_column]._int2str
        texts = [record[text_column] for record in train_dataset]
        labels = [id2label[record[label_column]] for record in train_dataset]
        X_train, X_dev, y_train, y_dev = train_test_split(texts, labels, test_size=self.train_test_split,
                                                          random_state=self.seed, stratify=labels, shuffle=True)

        test_dataset = raw_dataset['test']
        id2label = test_dataset.features[label_column]._int2str
        X_test = [record[text_column] for record in test_dataset]
        y_test = [id2label[record[label_column]] for record in test_dataset]

        train = pd.DataFrame({"text": X_train, "label": y_train})
        test = pd.DataFrame({"text": X_test, "label": y_test})
        dev = pd.DataFrame({"text": X_dev, "label": y_dev})

        return train, test, dev

    def climate_specificity(self):
        raw_dataset = load_dataset("climatebert/" + "climate_specificity")
        return self.train_test_huggingface_datasets(raw_dataset)

    def climate_sentiment(self):
        raw_dataset = load_dataset("climatebert/" + "climate_sentiment")
        return self.train_test_huggingface_datasets(raw_dataset)

    def climate_commitments_actions(self):
        raw_dataset = load_dataset("climatebert/" + "climate_commitments_actions")
        return self.train_test_huggingface_datasets(raw_dataset)

    def climate_detection(self):
        raw_dataset = load_dataset("climatebert/" + "climate_detection")
        return self.train_test_huggingface_datasets(raw_dataset)

    def climate_tcfd_recommendations(self):
        """
        'governance', 'metrics', 'none', 'risk', 'strategy'
        """
        raw_dataset = load_dataset("climatebert/" + "tcfd_recommendations")
        return self.train_test_huggingface_datasets(raw_dataset)

    def climatext(self):
        """
        TODO: use all the datasets ? combine the datasets or not because there is already a lot of data in the wikipedia dataset

        0,No
        1,Yes
        """
        folder_path = os.path.join(os.getcwd(), "data", "climaText")

        train_path = os.path.join(folder_path, "train", "Wiki-Doc-Train.tsv")
        train = pd.read_csv(train_path, sep="\t")[['label', 'sentence']]
        train.rename(columns={'sentence': 'text'}, inplace=True)

        test_path = os.path.join(folder_path, "test", "Wiki-Doc-Test.tsv")
        test = pd.read_csv(test_path, sep="\t")[['label', 'sentence']]
        test.rename(columns={'sentence': 'text'}, inplace=True)

        dev_path = os.path.join(folder_path, "dev", "Wiki-Doc-Dev.tsv")
        dev = pd.read_csv(dev_path, sep="\t")[['label', 'sentence']]
        dev.rename(columns={'sentence': 'text'}, inplace=True)

        # test_10k = pd.read_csv(self.path  + "/climaText/test/10-Ks (2018, test).tsv", sep="\t")[['label', 'sentence']]
        # test_10k.rename(columns={'sentence': 'text'}, inplace=True)
        #
        # test_claims = pd.read_csv(self.path  + "/climaText/test/Claims (test).tsv", sep="\t")[['label', 'sentence']]
        # test_claims.rename(columns={'sentence': 'text'}, inplace=True)

        return train, test, dev

    def climatext_wiki(self):
        train = pd.read_csv("data/climaText/train/AL-Wiki (train).tsv", sep="\t")
        dev = pd.read_csv("data/climaText/dev/Wikipedia (dev).tsv", sep="\t")
        test = pd.read_csv("data/climaText/test/Wikipedia (test).tsv", sep="\t")

        train = train[['label', 'sentence']]
        train.rename(columns={'sentence': 'text'}, inplace=True)

        test = test[['label', 'sentence']]
        test.rename(columns={'sentence': 'text'}, inplace=True)

        dev = dev[['label', 'sentence']]
        dev.rename(columns={'sentence': 'text'}, inplace=True)

        return train, test, dev

    def climatext_10k(self):
        train = pd.concat(
            [pd.read_csv("data/climaText/train/AL-Wiki (train).tsv", sep="\t")[['label', 'sentence']],
             pd.read_csv("data/climaText/train/AL-10Ks.tsv 3000 (58 positives, 2942 negatives) (TSV, 127138 KB).tsv", sep="\t")[['label', 'sentence']]
             ], ignore_index=True)

        dev = pd.read_csv("data/climaText/dev/Wikipedia (dev).tsv", sep="\t")

        test = pd.read_csv("data/climaText/test/10-Ks (2018, test).tsv", sep="\t")

        train = train[['label', 'sentence']]
        train.rename(columns={'sentence': 'text'}, inplace=True)

        test = test[['label', 'sentence']]
        test.rename(columns={'sentence': 'text'}, inplace=True)

        dev = dev[['label', 'sentence']]
        dev.rename(columns={'sentence': 'text'}, inplace=True)

        return train, test, dev

    def climatext_claim(self):
        train = pd.concat(
            [pd.read_csv("data/climaText/train/AL-Wiki (train).tsv", sep="\t")[['label', 'sentence']],
             pd.read_csv("data/climaText/train/AL-10Ks.tsv 3000 (58 positives, 2942 negatives) (TSV, 127138 KB).tsv",
                         sep="\t")[['label', 'sentence']]
             ], ignore_index=True)

        dev = pd.read_csv("data/climaText/dev/Wikipedia (dev).tsv", sep="\t")

        test = pd.read_csv("data/climaText/test/Claims (test).tsv", sep="\t")

        train = train[['label', 'sentence']]
        train.rename(columns={'sentence': 'text'}, inplace=True)

        test = test[['label', 'sentence']]
        test.rename(columns={'sentence': 'text'}, inplace=True)

        dev = dev[['label', 'sentence']]
        dev.rename(columns={'sentence': 'text'}, inplace=True)

        return train, test, dev

    def environmental_claims(self):
        """
        0,No
        1,Yes
        """
        folder_path = os.path.join(os.getcwd(), "data", "Environmental_claims")

        train_path = os.path.join(folder_path, "train.jsonl")
        train = pd.read_json(train_path, lines=True)
        test_path = os.path.join(folder_path, "test.jsonl")
        test = pd.read_json(test_path, lines=True)
        dev_path = os.path.join(folder_path, "dev.jsonl")
        dev = pd.read_json(dev_path, lines=True)

        return train, test, dev

    def ClimaTOPIC(self):
        """
        'Adaptation', 'Buildings', 'Climate Hazards', 'Emissions', 'Energy', 'Food', 'Governance and Data Management', 'Opportunities', 'Strategy', 'Transport', 'Waste', 'Water'
        """
        folder_path = os.path.join(os.getcwd(), "data", "Climabench", "ClimaTOPIC")

        train_path = os.path.join(folder_path, "train.csv")
        train = pd.read_csv(train_path, usecols=['Text', 'Label'])
        train.rename(columns={'Text': 'text', 'Label': 'label'}, inplace=True)

        test_path = os.path.join(folder_path, "test.csv")
        test = pd.read_csv(test_path, usecols=['Text', 'Label'])
        test.rename(columns={'Text': 'text', 'Label': 'label'}, inplace=True)

        dev_path = os.path.join(folder_path, "val.csv")
        dev = pd.read_csv(dev_path, usecols=['Text', 'Label'])
        dev.rename(columns={'Text': 'text', 'Label': 'label'}, inplace=True)

        return train, test, dev

    def climateFEVER_claim(self):
        """
        'DISPUTED', 'NOT_ENOUGH_INFO', 'REFUTES', 'SUPPORTS'
        """

        folder_path = os.path.join(os.getcwd(), "data", "climate-FEVER")

        dataset_path = os.path.join(folder_path, "climate-fever-dataset-r1.jsonl")
        ds = pd.read_json(dataset_path, lines=True)
        texts = ds['claim'].values
        labels = ds['claim_label'].values

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=self.train_test_split, random_state=self.seed, stratify=labels, shuffle=True)
        X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=self.test_dev_split, random_state=self.seed, stratify=y_test, shuffle=True)

        train = pd.DataFrame({"text": X_train, "label": y_train})
        test = pd.DataFrame({"text": X_test, "label": y_test})
        dev = pd.DataFrame({"text": X_dev, "label": y_dev})

        return train, test, dev

    def climateFEVER_evidence(self):
        folder_path = os.path.join(os.getcwd(), "data", "climate-FEVER")

        dataset_path = os.path.join(folder_path, "climate-fever-dataset-r1.jsonl")
        ds = pd.read_json(dataset_path, lines=True)

        #Todo: use sklearn train_test_split to add stratify
        train_ds = ds.sample(frac=1-self.train_test_split, random_state=self.seed).copy()
        dev_test_ds = ds.drop(train_ds.index).copy()
        test_ds = dev_test_ds.sample(frac=1-self.test_dev_split, random_state=self.seed)
        dev_ds = dev_test_ds.drop(test_ds.index).copy()

        train_ds = train_ds.explode('evidences')
        train_ds['text'] = train_ds['evidences'].apply(lambda x: x['evidence'])
        train_ds['label'] = train_ds['evidences'].apply(lambda x: x['evidence_label'])
        train_ds['hypothesis'] = train_ds['claim']

        test_ds = test_ds.explode('evidences')
        test_ds['text'] = test_ds['evidences'].apply(lambda x: x['evidence'])
        test_ds['label'] = test_ds['evidences'].apply(lambda x: x['evidence_label'])
        test_ds['hypothesis'] = test_ds['claim']

        dev_ds = dev_ds.explode('evidences')
        dev_ds['text'] = dev_ds['evidences'].apply(lambda x: x['evidence'])
        dev_ds['label'] = dev_ds['evidences'].apply(lambda x: x['evidence_label'])
        dev_ds['hypothesis'] = dev_ds['claim']

        train_ds.rename(columns={"text": "text", "hypothesis":"query"}, inplace=True)
        test_ds.rename(columns={"text": "text", "hypothesis":"query"}, inplace=True)
        dev_ds.rename(columns={"text": "text", "hypothesis":"query"}, inplace=True)

        return train_ds, test_ds, dev_ds

    def climateFEVER_evidence_climabench(self):
        folder_path = os.path.join(os.getcwd(), "data", "climate-FEVER")

        dataset_path = os.path.join(folder_path, "climate-fever-dataset-r1.jsonl")
        ds = pd.read_json(dataset_path, lines=True)

        ds = ds.explode('evidences')
        ds['text'] = ds['evidences'].apply(lambda x: x['evidence'])
        ds['label'] = ds['evidences'].apply(lambda x: x['evidence_label'])
        ds['hypothesis'] = ds['claim']

        ds.reset_index(drop=True, inplace=True)

        # Todo: use sklearn train_test_split to add stratify
        train_ds = ds.sample(frac=1 - self.train_test_split, random_state=self.seed).copy()
        dev_test_ds = ds.drop(train_ds.index).copy()
        test_ds = dev_test_ds.sample(frac=1 - self.test_dev_split, random_state=self.seed)
        dev_ds = dev_test_ds.drop(test_ds.index).copy()

        train_ds.rename(columns={"text": "text", "hypothesis": "query"}, inplace=True)
        test_ds.rename(columns={"text": "text", "hypothesis": "query"}, inplace=True)
        dev_ds.rename(columns={"text": "text", "hypothesis": "query"}, inplace=True)

        return train_ds, test_ds, dev_ds

    def climateBUG_data(self):
        # TODO: ClimateBUG has a train and a test dataset
        # TODO: the dataset has a manual column, but it seems to be always true
        raw_dataset = load_dataset("lumilogic/climateBUG-Data", token=os.environ["HUB_TOKEN"])
        return self.train_test_huggingface_datasets(raw_dataset, text_column="statement")

    def sciDCC(self):
        """
        All columns from the dataset that might be used: 'Date', 'Link', 'Title', 'Summary', 'Body', 'Category', 'Year'
        They don't give recommendations in the paper, however, after checking the label distribution in time, we decide to use a stratify instead of a time split.
        Using a time split resulted in a train dataset missing 1 label and the test dataset missing 4 labels.

        'Agriculture & Food', 'Animals', 'Biology', 'Biotechnology', 'Climate', 'Earthquakes', 'Endangered Animals', 'Environment', 'Extinction', 'Genetically Modified', 'Geography', 'Geology', 'Hurricanes Cyclones', 'Microbes', 'New Species', 'Ozone Holes', 'Pollution', 'Weather', 'Zoology'

        :return:
        """
        folder_path = os.path.join(os.getcwd(), "data", "SciDCC")

        dataset_path = os.path.join(folder_path, "SciDCC.csv")
        raw_dataset = pd.read_csv(dataset_path, usecols=['Summary', 'Category', 'Date'])
        raw_dataset['Date'] = pd.to_datetime(raw_dataset['Date'])

        raw_dataset.rename(columns={
            "Summary":"text",
            "Category":"label"
        }, inplace=True)

        train, test, dev = split_stratify_time(
            df=raw_dataset,
            test_size=self.train_test_split,
            dev_size=self.test_dev_split
        )

        return train, test, dev

    def contrarian_claims(self):
        """
        0.0 No claim, No claim

        1.1,Global warming is not happening,Ice/permafrost/snow cover isn’t melting
        1.2,Global warming is not happening,We’re heading into an ice age/global cooling
        1.3,Global warming is not happening,Weather is cold/snowing
        1.4,Global warming is not happening,Climate hasn’t warmed/changed over the last (few) decade(s)
        1.6,Global warming is not happening,Sea level rise is exaggerated/not accelerating
        1.7,Global warming is not happening,Extreme weather isn’t increasing/has happened before/isn’t linked to climate change

        2.1,Human greenhouse gases are not causing climate change,It’s natural cycles/variation
        2.3,Human greenhouse gases are not causing climate change,There’s no evidence for greenhouse effect/carbon dioxide driving climate change

        3.1,Climate impacts/global warming is beneficial/not bad,Climate sensitivity is low/negative feedbacks reduce warming
        3.2,Climate impacts/global warming is beneficial/not bad,Species/plants/reefs aren’t showing climate impacts/are benefiting from climate change
        3.3,Climate impacts/global warming is beneficial/not bad,CO2 is beneficial/not a pollutant

        4.1,Climate solutions won’t work,Climate policies (mitigation or adaptation) are harmful
        4.2,Climate solutions won’t work,Climate policies areineffective/flawed
        4.4,Climate solutions won’t work,Clean energy technology/biofuels won’t work
        4.5,Climate solutions won’t work,People need energy (e.g. from fossil fuels/nuclear)

        5.1,Climate movement/science is unreliable,Climate-related science is unreliable/uncertain/unsound (data, methods & models)
        5.2,Climate movement/science is unreliable,Climate movement is unreliable/alarmist/corrupt
        """
        folder_path = os.path.join(os.getcwd(), "data", "Coan_contrarian_claims", "training")

        train_path = os.path.join(folder_path, "training.csv")
        train = pd.read_csv(train_path)
        train.columns = ["text", "label"]

        test_path = os.path.join(folder_path, "test.csv")
        test = pd.read_csv(test_path)
        test.columns = ["text", "label"]
        test = test[~test["text"].isna()].copy()

        dev_path = os.path.join(folder_path, "validation.csv")
        dev = pd.read_csv(dev_path)
        dev.columns = ["text", "label"]

        return train, test, dev

    def green_claims(self):
        """
        'green_claim', 'not_green'
        """
        folder_path = os.path.join(os.getcwd(), "data", "woloszyn_green_claims")
        dataset_path = os.path.join(folder_path, "green_claims.csv")
        raw_dataset = pd.read_csv(dataset_path,
                                  usecols=["tweet", "label_binary"])
        raw_dataset.columns=["text", "label"]

        train, temp = train_test_split(raw_dataset, test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev


    def green_claims_3(self):
        """
        labels: 'implicit_claim', 'explicit_claim', 'not_green'
        """
        folder_path = os.path.join(os.getcwd(), "data", "woloszyn_green_claims")
        dataset_path = os.path.join(folder_path, "green_claims.csv")

        raw_dataset = pd.read_csv(dataset_path,
                                  usecols=["tweet", "label_multi"])
        raw_dataset.columns=["text", "label"]

        train, temp = train_test_split(raw_dataset, test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev


    # Task 1: Page contains evidence (yes/no)
    def lobbymap_pages(self):
        """
        TODO: the dataset is quite complicated, i should create a page level dataset, then we need to correct evaluation metric
        In the paper, they reconstruct the pages by concatenating the sentences.
        :return:
        """
        folder_path = os.path.join(os.getcwd(), "data", "lobbymap", "lobbymap_dataset")

        train_path = os.path.join(folder_path, "train.jsonl")
        raw_train = pd.read_json(train_path, lines=True)

        test_path = os.path.join(folder_path, "test.jsonl")
        raw_test = pd.read_json(test_path, lines=True)

        dev_path = os.path.join(folder_path, "valid.jsonl")
        raw_dev = pd.read_json(dev_path, lines=True)

        def page_dataset(raw):
            pages = reconstruct_page(raw.copy())
            raw['pages_evidences_label'] = raw['evidences'].apply(get_page_idx)
            pages = pages.merge(raw[['document_id', 'pages_evidences_label']], on='document_id', how='left')
            pages['label'] = pages[['page_idx', 'pages_evidences_label']].apply(lambda x: x['page_idx'] in x['pages_evidences_label'],  axis=1)
            pages['label'] = 1*pages['label']
            return pages[['text', 'label']].copy()

        train = page_dataset(raw_train)
        test = page_dataset(raw_test)
        dev = page_dataset(raw_dev)

        return train, test, dev

    # Task 2: Given a page, multilabel classification for the page query
    def lobbymap_query(self, relation=False):
        """
        **Multi-label classification**

        :return:
        """
        folder_path = os.path.join(os.getcwd(), "data", "lobbymap", "lobbymap_dataset")

        train_path = os.path.join(folder_path, "train.jsonl")
        raw_train = pd.read_json(train_path, lines=True)

        test_path = os.path.join(folder_path, "test.jsonl")
        raw_test = pd.read_json(test_path, lines=True)

        dev_path = os.path.join(folder_path, "valid.jsonl")
        raw_dev = pd.read_json(dev_path, lines=True)

        def query_dataset(raw):
            map_query_label = get_page_query_map(raw)
            pages = reconstruct_page(raw.copy())
            query = map_query_label.merge(pages, left_on=['document_id', 'page_indices'], right_on=['document_id', 'page_idx'], how='left')
            query = query[['text', 'query']].copy()
            query.columns=['text', 'label']
            return query[['text', 'label']]

        train = query_dataset(raw_train)
        test = query_dataset(raw_test)
        dev = query_dataset(raw_dev)

        if relation:
            train_exploded = train.explode("label")
            labels = train_exploded['label'].unique()
            categories_df = pd.DataFrame(labels, columns=["query"])

            train = train.merge(categories_df, how='cross')
            train['category_in_label'] = train.apply(lambda row: row['query'] in row['label'], axis=1)
            test = test.merge(categories_df, how='cross')
            test['category_in_label'] = test.apply(lambda row: row['query'] in row['label'], axis=1)
            dev = dev.merge(categories_df, how='cross')
            dev['category_in_label'] = dev.apply(lambda row: row['query'] in row['label'], axis=1)

            train = train[['text', 'query', 'category_in_label']]
            train.rename(columns={'category_in_label': 'label'}, inplace=True)
            test = test[['text', 'query', 'category_in_label']]
            test.rename(columns={'category_in_label': 'label'}, inplace=True)
            dev = dev[['text', 'query', 'category_in_label']]
            dev.rename(columns={'category_in_label': 'label'}, inplace=True)

            return train, test, dev
        else:
            return train, test, dev

    # def get_lobbymap_queries(self):
    #     return os.listdir(os.path.join(os.getcwd(), "data", "lobbymap", "lobbymap_dataset", "multilabel_datasets"))
    #
    # def lobbymap_query_multilabel(self, label):
    #     """
    #     **Multi-label classification**
    #
    #     :return:
    #     """
    #     folder_path = os.path.join(os.getcwd(), "data", "lobbymap", "lobbymap_dataset", "multilabel_datasets", label)
    #
    #     train_path = os.path.join(folder_path, "train.csv")
    #     raw_train = pd.read_csv(train_path)
    #
    #     test_path = os.path.join(folder_path, "test.csv")
    #     raw_test = pd.read_csv(test_path)
    #
    #     dev_path = os.path.join(folder_path, "dev.csv")
    #     raw_dev = pd.read_csv(dev_path)
    #
    #     return raw_train, raw_test, raw_dev

    def lobbymap_query(self):
        """
        **Multi-label classification**

        :return:
        """
        folder_path = os.path.join(os.getcwd(), "data", "lobbymap", "lobbymap_dataset")

        train_path = os.path.join(folder_path, "train_query.pkl")
        raw_train = pd.read_parquet(train_path)

        test_path = os.path.join(folder_path, "test_query.pkl")
        raw_test = pd.read_parquet(test_path)

        dev_path = os.path.join(folder_path, "dev_query.pkl")
        raw_dev = pd.read_parquet(dev_path)

        return raw_train, raw_test, raw_dev


    # Task 3: Given a page and a Query, predict the stances
    def lobbymap_stance(self):
        folder_path = os.path.join(os.getcwd(), "data", "lobbymap", "lobbymap_dataset")

        train_path = os.path.join(folder_path, "train.jsonl")
        raw_train = pd.read_json(train_path, lines=True)

        test_path = os.path.join(folder_path, "test.jsonl")
        raw_test = pd.read_json(test_path, lines=True)

        dev_path = os.path.join(folder_path, "valid.jsonl")
        raw_dev = pd.read_json(dev_path, lines=True)

        def binary_stance_dataset(raw):
            map_stance_label = get_page_stance_map(raw)
            pages = reconstruct_page(raw.copy())
            stance = map_stance_label.merge(pages, left_on=['document_id', 'page_indices'],
                                          right_on=['document_id', 'page_idx'], how='left')
            stance = stance[['text', 'query', 'stance']].copy()
            stance.columns=['text', 'query', 'label']
            return stance

        train = binary_stance_dataset(raw_train)
        test = binary_stance_dataset(raw_test)
        dev = binary_stance_dataset(raw_dev)
        return train, test, dev

    def sustainable_signals_review(self):
        """
        TODO: check the aggregation method in the paper
        Review relevant to sustainability
        Labels: 'Not relevant', 'Relevant'
        :return:
        """
        folder_path = os.path.join(os.getcwd(), "data", "sustainablesignals")
        dataset_path = os.path.join(folder_path, "amazon product sus annotations.csv")

        raw_dataset = pd.read_csv(dataset_path)

        annotator_columns = ['annotator_1', 'annotator_2', 'annotator_3', 'annotator_4']
        raw_dataset['label'] = raw_dataset[annotator_columns].mode(axis=1)[0]
        raw_dataset.rename(columns={'review':'text'}, inplace=True)

        train, temp = train_test_split(raw_dataset[['text', 'label']], test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev

    def esgbert_datasets(self, dataset_name, target):
        raw_dataset = load_dataset("ESGBERT/"+dataset_name)['train']

        texts = [record['text'] for record in raw_dataset]
        labels = [record[target] for record in raw_dataset]

        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=self.train_test_split,
                                                            random_state=self.seed, stratify=labels, shuffle=True)
        X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=self.test_dev_split,
                                                        random_state=self.seed, stratify=y_test, shuffle=True)

        train = pd.DataFrame({"text": X_train, "label": y_train})
        test = pd.DataFrame({"text": X_test, "label": y_test})
        dev = pd.DataFrame({"text": X_dev, "label": y_dev})

        return train, test, dev

    def esgbert_e(self):
        """
        Topic: Environment
        Labels: Yes, No
        """
        return self.esgbert_datasets("environmental_2k", 'env')

    def esgbert_s(self):
        """
        Topic: Social
        Labels: Yes, No
        """
        return self.esgbert_datasets("social_2k", 'soc')

    def esgbert_g(self):
        """
        Topic: Governance
        Labels: Yes, No
        """
        return self.esgbert_datasets("governance_2k", 'gov')

    def esgbert_action500(self):
        """
        Topic: Forest
        Labels: Yes, No
        """
        return self.esgbert_datasets("action_500", 'action')

    def esgbert_category_water(self):
        """
        #TODO: change this into the multilabel task ?
        Topic: Water
        Labels: Yes, No
        """
        dataset_path = os.path.join(os.getcwd(), "data", "esgbert", "category", "WaterForestBiodiversityNature_2200.csv")

        raw_dataset = pd.read_csv(dataset_path)
        raw_dataset = raw_dataset[['text', 'Water']]
        raw_dataset.columns = ['text', 'label']

        train, temp = train_test_split(raw_dataset, test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev

    def esgbert_category_forest(self):
        """
        Topic: Forest
        Labels: Yes, No
        """
        dataset_path = os.path.join(os.getcwd(), "data", "esgbert", "category", "WaterForestBiodiversityNature_2200.csv")

        raw_dataset = pd.read_csv(dataset_path)
        raw_dataset = raw_dataset[['text', 'Forest']]
        raw_dataset.columns = ['text', 'label']

        train, temp = train_test_split(raw_dataset, test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev

    def esgbert_category_biodiversity(self):
        """
        Topic: Biodiversity
        Labels: Yes, No
        """
        dataset_path = os.path.join(os.getcwd(), "data", "esgbert", "category", "WaterForestBiodiversityNature_2200.csv")

        raw_dataset = pd.read_csv(dataset_path)
        raw_dataset = raw_dataset[['text', 'Biodiversity']]
        raw_dataset.columns = ['text', 'label']

        train, temp = train_test_split(raw_dataset, test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev

    def esgbert_category_nature(self):
        """
        Topic: Nature
        Labels: Yes, No
        """
        dataset_path = os.path.join(os.getcwd(), "data", "esgbert", "category", "WaterForestBiodiversityNature_2200.csv")

        raw_dataset = pd.read_csv(dataset_path)
        raw_dataset = raw_dataset[['text', 'Nature']]
        raw_dataset.columns = ['text', 'label']

        train, temp = train_test_split(raw_dataset, test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev

    def climateEng(self):
        """
        Topics of the discussion

        remove @blablabla ?

        {0:"General",
         1:"Politics",
         2:"Ocean/Water",
         3:"Agriculture/Forestry",
         4:"Disaster"}
         verified
        """
        # TODO: search for data from oana

        folder_path = os.path.join(os.getcwd(), "data", "Climabench", "ClimateEng")

        train_path = os.path.join(folder_path, "train.csv")
        raw_train = pd.read_csv(train_path)

        test_path = os.path.join(folder_path, "test.csv")
        raw_test = pd.read_csv(test_path)

        dev_path = os.path.join(folder_path, "val.csv")
        raw_dev = pd.read_csv(dev_path)

        return raw_train, raw_test, raw_dev

    def climateStance(self):
        """
            {
            1:"Favor",# 1.0
            0:"Ambiguous", # 0.0
            2:"Against", #-1.0
            }
        verified
        :return:
        """
        # TODO: search for data from oana

        folder_path = os.path.join(os.getcwd(), "data", "Climabench", "ClimateStance")

        train_path = os.path.join(folder_path, "train.csv")
        raw_train = pd.read_csv(train_path)

        test_path = os.path.join(folder_path, "test.csv")
        raw_test = pd.read_csv(test_path)

        dev_path = os.path.join(folder_path, "val.csv")
        raw_dev = pd.read_csv(dev_path)

        return raw_train, raw_test, raw_dev

    def climateEng_Reddit(self):
        """
        Topics of the discussion
        1. Disaster
        2. Ocean/Water
        3. Agriculture/Forestry
        4. Politics
        5. General
        """
        folder_path = os.path.join(os.getcwd(), "data", "vaid_climate_twitter", "ClimateReddit")


        first_annotator = pd.read_csv(os.path.join(folder_path, "Reddit-Annotator1.tsv"), sep="\t")
        second_annototar = pd.read_csv(os.path.join(folder_path, "Reddit-Annotator2.tsv"), sep="\t")

        raw_dataset = first_annotator.merge(second_annototar, on='permalink', suffixes=('_1', '_2'))
        raw_dataset = raw_dataset[raw_dataset['topic-label_1'] == raw_dataset['topic-label_2']]
        raw_dataset = raw_dataset[['body_1', 'topic-label_1']].rename(
            columns={'body_1': 'text', 'topic-label_1': 'label'})

        train, temp = train_test_split(raw_dataset, test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev

    def climateStance_Reddit(self):
        """
        stance on global warming
        'neutral', 'disagrees', 'agrees'
        """
        folder_path = os.path.join(os.getcwd(), "data", "vaid_climate_twitter", "ClimateReddit")

        first_annotator = pd.read_csv(os.path.join(folder_path, "Reddit-Annotator1.tsv"), sep="\t")
        second_annototar = pd.read_csv(os.path.join(folder_path, "Reddit-Annotator2.tsv"), sep="\t")

        raw_dataset = first_annotator.merge(second_annototar, on='permalink', suffixes=('_1', '_2'))
        raw_dataset = raw_dataset[raw_dataset['opinion-label_1'] == raw_dataset['opinion-label_2']]
        raw_dataset = raw_dataset[['body_1', 'opinion-label_1']].rename(
            columns={'body_1': 'text', 'opinion-label_1': 'label'})

        train, temp = train_test_split(raw_dataset, test_size=self.train_test_split, random_state=self.seed, stratify=raw_dataset['label'], shuffle=True)
        dev, test = train_test_split(temp, test_size=self.test_dev_split, random_state=self.seed, stratify=temp['label'], shuffle=True)

        return train, test, dev

    def climateStance_gwsd(self):
        """
        stance on global warming
        'neutral', 'disagrees', 'agrees'
        """
        dataset_path = os.path.join(os.getcwd(), "data", "gwsd", "GWSD.tsv")
        raw_dataset = pd.read_csv(dataset_path, sep="\t")

        def majority_label(row):
            worker_columns = [f"worker_{i}" for i in range(8)]
            labels = row[worker_columns]
            return labels.mode()[0]

        raw_dataset['gold_label'] = raw_dataset.apply(majority_label, axis=1)

        test = raw_dataset[raw_dataset['in_held_out_test'] == True]
        test = test[['sentence', 'gold_label']]
        test.rename(columns={'sentence': 'text', 'gold_label': 'label'}, inplace=True)

        train_dev = raw_dataset[raw_dataset['in_held_out_test'] == False]
        train_dev = train_dev[['sentence', 'gold_label']]
        train_dev.rename(columns={'sentence': 'text', 'gold_label': 'label'}, inplace=True)

        train, dev = train_test_split(train_dev, test_size=self.train_test_split * self.test_dev_split, random_state=self.seed, stratify=train_dev['label'], shuffle=True) #TODO: see if i split differently

        return train, test, dev

    def logicClimate(self):
        """
        fallacies in climate related text
        'intentional', 'fallacy of credibility', 'false dilemma'
        'appeal to emotion', 'equivocation', 'faulty generalization'
        'fallacy of relevance', 'fallacy of logic', 'ad populum', 'false causality'
        'ad hominem', 'fallacy of extension', 'circular reasoning'
        """
        #Todo: Trian on larger dataset using first the cLogic dataset, then the climatelogic ?

        folder_path = os.path.join(os.getcwd(), "data", "logicclimate")


        train_path = os.path.join(folder_path, "multi_train.csv")
        train = pd.read_csv(train_path)[['source_article', 'logical_fallacies']]
        train.rename(columns={'source_article': 'text', 'logical_fallacies': 'label'}, inplace=True)

        test_path = os.path.join(folder_path, "multi_test.csv")
        test = pd.read_csv(test_path)[['source_article', 'logical_fallacies']]
        test.rename(columns={'source_article': 'text', 'logical_fallacies': 'label'}, inplace=True)

        dev_path = os.path.join(folder_path, "multi_dev.csv")
        dev = pd.read_csv(dev_path)[['source_article', 'logical_fallacies']]
        dev.rename(columns={'source_article': 'text', 'logical_fallacies': 'label'}, inplace=True)

        return train, test, dev

    def climaInsurance(self):
        """
        questions from NAIC (insurance):
            1. EMISSIONS Does the company have a plan to assess, reduce or mitigate
            its emissions in its operations or organizations? If yes, please summarize.
            2. RISK PLAN Does the company have a climate change policy with respect
            to risk management and investment management? If yes, please
            summarize. If no, how do you account for climate change in your risk
            management?
            3. ASSESS Describe your company’s process for identifying climate change-
            related risks and assessing the degree that they could affect your
            business, including financial implications.
            4. RISKS Summarize the current or anticipated risks that climate change
            poses to your company. Explain the ways that these risks could affect
            your business. Include identification of the geographical areas affected
            by these risks.
            5. INVEST Part A: Has the company considered the impact of climate change
            on its investment portfolio? Part B: Has it altered its investment strategy
            in response to these considerations? If so, please summarize steps you
            have taken.
            6. MITIGATE Summarize steps the company has taken to encourage
            policyholders to reduce the losses caused by climate change-influenced
            events.
            7. ENGAGE Discuss steps, if any, the company has taken to engage key
            constituencies on the topic of climate change.
            8. MANAGE Describe actions the company is taking to manage the risks
            climate change poses to your business including, in general terms,
            the use of computer modeling. If Yes – Please summarize what actions
            the company is taking and in general terms the use if any of computer
            modeling in response text box

            I think i won't include the binary classification

            order of questions ?
        """
        folder_path = os.path.join(os.getcwd(), "data", "Climabench", "ClimateInsuranceMulti")

        train_path = os.path.join(folder_path, "train.csv")
        train = pd.read_csv(train_path)[['text', "label"]]

        test_path = os.path.join(folder_path, "test.csv")
        test = pd.read_csv(test_path)[['text', "label"]]

        dev_path = os.path.join(folder_path, "val.csv")
        dev = pd.read_csv(dev_path)[['text', "label"]]

        return train, test, dev

    def climaInsuranceOurs(self):
        """
        questions from NAIC (insurance):
            1. EMISSIONS Does the company have a plan to assess, reduce or mitigate
            its emissions in its operations or organizations? If yes, please summarize.
            2. RISK PLAN Does the company have a climate change policy with respect
            to risk management and investment management? If yes, please
            summarize. If no, how do you account for climate change in your risk
            management?
            3. ASSESS Describe your company’s process for identifying climate change-
            related risks and assessing the degree that they could affect your
            business, including financial implications.
            4. RISKS Summarize the current or anticipated risks that climate change
            poses to your company. Explain the ways that these risks could affect
            your business. Include identification of the geographical areas affected
            by these risks.
            5. INVEST Part A: Has the company considered the impact of climate change
            on its investment portfolio? Part B: Has it altered its investment strategy
            in response to these considerations? If so, please summarize steps you
            have taken.
            6. MITIGATE Summarize steps the company has taken to encourage
            policyholders to reduce the losses caused by climate change-influenced
            events.
            7. ENGAGE Discuss steps, if any, the company has taken to engage key
            constituencies on the topic of climate change.
            8. MANAGE Describe actions the company is taking to manage the risks
            climate change poses to your business including, in general terms,
            the use of computer modeling. If Yes – Please summarize what actions
            the company is taking and in general terms the use if any of computer
            modeling in response text box

            I think i won't include the binary classification

            order of questions ?
        """
        folder_path = os.path.join(os.getcwd(), "data", "green_nlp", "ClimaINS")

        train_path = os.path.join(folder_path, "train.pkl")
        train = pd.read_parquet(train_path)

        test_path = os.path.join(folder_path, "test.pkl")
        test = pd.read_parquet(test_path)

        dev_path = os.path.join(folder_path, "dev.pkl")
        dev = pd.read_parquet(dev_path)

        return train, test, dev


    def climaQA(self):
        """
        The dataset contains duplicates with label 1 and 0 for the same pair. We remove the 0.
        We guess that the dataset was constructed through a cartesian product.

        1,Answer the question
        0,Do not answer the question
        """
        folder_path = os.path.join(os.getcwd(), "data", "Climabench", "ClimaQA", "corpo")

        train_path = os.path.join(folder_path, "train_qa.csv")
        raw_train = pd.read_csv(train_path)
        raw_train.rename(columns={"question": "query", "answer":"text"}, inplace=True)
        duplicates_train = raw_train[raw_train[['query', 'text']].duplicated(keep=False)]
        to_be_dropped = duplicates_train[duplicates_train['label'] == 0]
        raw_train = raw_train.drop(to_be_dropped.index)

        test_path = os.path.join(folder_path, "test_qa.csv")
        raw_test = pd.read_csv(test_path)
        raw_test.rename(columns={"question": "query", "answer":"text"}, inplace=True)
        duplicates_test = raw_test[raw_test[['query', 'text']].duplicated(keep=False)]
        to_be_dropped = duplicates_test[duplicates_test['label'] == 0]
        raw_test = raw_test.drop(to_be_dropped.index)

        dev_path = os.path.join(folder_path, "val_qa.csv")
        raw_dev = pd.read_csv(dev_path)
        raw_dev.rename(columns={"question": "query", "answer":"text"}, inplace=True)
        duplicates_dev = raw_dev[raw_dev[['query', 'text']].duplicated(keep=False)]
        to_be_dropped = duplicates_dev[duplicates_dev['label'] == 0]
        raw_dev = raw_dev.drop(to_be_dropped.index)

        return raw_train, raw_test, raw_dev

    def OurClimaQA(self):
        """
        The dataset contains duplicates with label 1 and 0 for the same pair. We remove the 0.
        We guess that the dataset was constructed through a cartesian product.

        1,Answer the question
        0,Do not answer the question
        """
        folder_path = os.path.join(os.getcwd(), "data", "Climabench", "ClimaQA", "CustomDataset")

        train_path = os.path.join(folder_path, "train.pkl")
        raw_train = pd.read_parquet(train_path)

        test_path = os.path.join(folder_path, "test.pkl")
        raw_test = pd.read_parquet(test_path)

        dev_path = os.path.join(folder_path, "dev.pkl")
        raw_dev = pd.read_parquet(dev_path)

        return raw_train, raw_test, raw_dev

    def filter(self, dataset_df, max_token=4000, min_token=5, drop=True, query=False):

        print(dataset_df.columns)

        dataset_df = dataset_df[dataset_df['token_counts'] < max_token]
        dataset_df = dataset_df[dataset_df['token_counts'] >= min_token]

        if drop == True:
            dataset_df = dataset_df.drop('token_counts', axis=1)
            dataset_df = dataset_df.drop('language', axis=1)
            dataset_df = dataset_df.drop('gibberish', axis=1)
            dataset_df = dataset_df.drop('text', axis=1)
            dataset_df.rename(columns={"clean_text":"text"}, inplace=True)

        return dataset_df

    def prepare_filter(self, dataset_df):
        dataset_df['clean_text'] = dataset_df['text'].apply(clean_text)
        dataset_df['token_counts'] = dataset_df['clean_text'].apply(self.count_tokens)

        return dataset_df

    def weighted_random_sampling(self, data, label_column, n_samples):
        label_distrib = data[label_column].value_counts()
        label_distrib.sort_values(ascending=True, inplace=True)

        sampled_data = []
        n_label = len(label_distrib)

        n = 0
        i = 0
        for label in label_distrib.index:
            label_data = data[data[label_column] == label]

            N_target = int((n_samples - n) / (n_label - i))

            if len(label_data) < N_target:
                sampled_data += [label_data]

                n += len(label_data)
            else:
                sample = label_data.sample(N_target, random_state=self.seed)
                sampled_data += [sample]
                n += len(sample)

            i += 1

        return pd.concat(sampled_data)

    def rebalance(self, df):
        rus = RandomUnderSampler(random_state=self.seed)
        df, _ = rus.fit_resample(X=df, y=df[['label']])

        return df

    def resize(self, df, max_size, stratify_on="label"):
        if max_size >= len(df):
            return df
        else:
            return resample(
                df,
                replace=False,
                n_samples=max_size,
                random_state=self.seed,
                stratify=df[stratify_on]
            )

    def truncate(self, train, dev, max_size=10000, balanced="balanced", stratify_on="label"):
        """
        Truncate the dataset to reduce its size.

        :param train: training dataset (pandas DataFrame)
        :param dev: validation/dev dataset (pandas DataFrame)
        :param max_size: Limit the number of examples in the dataset (int)
        :param max_size_per_label: Limit the number of examples per label (int)
        :param balanced: False to keep the original distribution and True to balance the labels (bool)
        :return: truncated train and dev datasets (tuple of pandas DataFrames)
        """

        if balanced=="random":
            train = self.rebalance(train)
            dev = self.rebalance(dev)

            train = self.resize(train, max_size, stratify_on)
            dev = self.resize(dev, max_size, stratify_on)
        elif balanced=="weighted":
            train = self.weighted_random_sampling(train, 'label', max_size)

            if len(dev) > max_size:
                dev = self.weighted_random_sampling(dev, 'label', max_size)
        else:
            train = self.resize(train, max_size, stratify_on)
            dev = self.resize(dev, max_size, stratify_on)


        return train, dev

    def get_inputs_names(self, dataset_name):
        if dataset_name in self.relation_datasets.keys():
            return ['text', 'query']
        if dataset_name in self.stance_datasets.keys():
            return ['text', 'query']
        else:
            return 'text'


