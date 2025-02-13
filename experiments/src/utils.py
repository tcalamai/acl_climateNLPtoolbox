from experiment import load_dataset, generate_args, clean_datasets
from src.builder import DatasetBuilder
from src.logger import Logger
import pandas as pd
import os

class Generator():
    def __init__(self):
        self.dataset_builder = DatasetBuilder(seed=42, load_tokenizer=False)
        self.dataset_list = list(self.dataset_builder.datasets.keys()) + list(self.dataset_builder.stance_datasets.keys()) + list(self.dataset_builder.relation_datasets.keys()) + list(self.dataset_builder.multilabel_datasets.keys())
        self.logger = Logger("subset_log")
        self.args = generate_args(self.dataset_builder, self.dataset_list, self.logger)
        self.dataset_max_size = 10000
        self.seed = 42

    def load_dataset(self, dataset_name):
        dataset_path = os.path.join(os.getcwd(), "data", "green_nlp_tasks", dataset_name)
        train = pd.read_parquet(os.path.join(dataset_path, "train.pkl"))
        test = pd.read_parquet(os.path.join(dataset_path, "test.pkl"))
        dev = pd.read_parquet(os.path.join(dataset_path, "dev.pkl"))

        out = dict()

        out['X_train'] = train[self.args[dataset_name]["input_columns"]]  # .head(50)
        out['y_train'] = train[self.args[dataset_name]['label_columns']]  # .head(50)

        out['X_test'] = test[self.args[dataset_name]["input_columns"]]  # .head(50)
        out['y_test'] = test[self.args[dataset_name]['label_columns']]  # .head(50)

        out['X_dev'] = dev[self.args[dataset_name]["input_columns"]]  # .head(50)
        out['y_dev'] = dev[self.args[dataset_name]['label_columns']]  # .head(50)

        return train, test, dev, out

    def load_nlp_tasks_dataset(self, dataset_name):
        dataset_path = os.path.join(os.getcwd(), "data", "green_nlp_tasks", dataset_name)
        train = pd.read_parquet(os.path.join(dataset_path, "train.pkl"))
        test = pd.read_parquet(os.path.join(dataset_path, "test.pkl"))
        dev = pd.read_parquet(os.path.join(dataset_path, "dev.pkl"))

        out = dict()

        out['X_train'] = train[self.args[dataset_name]["input_columns"]]  # .head(50)
        out['y_train'] = train[self.args[dataset_name]['label_columns']]  # .head(50)

        out['X_test'] = test[self.args[dataset_name]["input_columns"]]  # .head(50)
        out['y_test'] = test[self.args[dataset_name]['label_columns']]  # .head(50)

        out['X_dev'] = dev[self.args[dataset_name]["input_columns"]]  # .head(50)
        out['y_dev'] = dev[self.args[dataset_name]['label_columns']]  # .head(50)

        return train, test, dev, out

    def load_dataset_unprocessed(self, dataset_name):
        train, test, dev = load_dataset(dataset_name)
        # train, test, dev = clean_datasets(train, test, dev, dataset_name, self.dataset_builder)

        return train, test, dev

    def loading_raw_datasets(self, dataset_name):
        dataset_path = os.path.join(os.getcwd(), "data", "cleaned_datasets", dataset_name)
        train = pd.read_parquet(os.path.join(dataset_path, "train.pkl"))
        test = pd.read_parquet(os.path.join(dataset_path, "test.pkl"))
        dev = pd.read_parquet(os.path.join(dataset_path, "dev.pkl"))

        return train, test, dev

    def loading_processed_dataset(self, dataset_name):
        dataset_path = os.path.join(os.getcwd(), "data", "cleaned_datasets", dataset_name)
        train = pd.read_parquet(os.path.join(dataset_path, "train.pkl"))
        test = pd.read_parquet(os.path.join(dataset_path, "test.pkl"))
        dev = pd.read_parquet(os.path.join(dataset_path, "dev.pkl"))

        train, test, dev = clean_datasets(train, test, dev, dataset_name, self.dataset_builder) # deterministic (drop column, filter)

        # determinisitc
        if isinstance(self.args[dataset_name]['input_columns'], list):
            train = train.drop_duplicates(subset = self.args[dataset_name]['input_columns'] + [self.args[dataset_name]['label_columns']])
            test = test.drop_duplicates(subset = self.args[dataset_name]['input_columns'] + [self.args[dataset_name]['label_columns']])
            dev = dev.drop_duplicates(subset = self.args[dataset_name]['input_columns'] + [self.args[dataset_name]['label_columns']])
        else:
            train = train.drop_duplicates(subset=[self.args[dataset_name]['input_columns'], self.args[dataset_name]['label_columns']])
            test = test.drop_duplicates(subset=[self.args[dataset_name]['input_columns'], self.args[dataset_name]['label_columns']])
            dev = dev.drop_duplicates(subset=[self.args[dataset_name]['input_columns'], self.args[dataset_name]['label_columns']])

        # deterministic (equilibrate climateFEVER_evidence)
        if dataset_name == "climateFEVER_evidence":
            print("Downsampling to have a balanced dataset")

            sampled_data = train[~(train['label'] == "NOT_ENOUGH_INFO")].copy()
            sampled_data = pd.concat([sampled_data,
                                      train[(train['label'] == "NOT_ENOUGH_INFO")].sample(1539, replace=False,
                                                                                          random_state=self.seed)])

            train = sampled_data


        if train.shape[0] > self.dataset_max_size:
            print(f"Dataset size ({train.shape[0]}) is larger than {self.dataset_max_size}: Truncating the dataset to {self.dataset_max_size}, with a balanced label distribution for training and evaluation datasets. (Test dataset remain the same)")
            train, dev = self.dataset_builder.truncate(train=train, dev=dev, max_size=self.dataset_max_size, balanced=self.args[dataset_name]["balanced"], stratify_on=self.args[dataset_name]["stratify_on"])
            #TODO: add an automatic check to see if the dataset is balanced ? use args ?

        return train, test, dev
