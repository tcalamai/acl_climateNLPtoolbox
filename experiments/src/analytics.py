from src.builder import DatasetBuilder
import os

import pandas as pd
import numpy as np


def f1_score_random(distribution):
    distribution = distribution['test'].values
    f1_scores = (1 / len(distribution)) * np.sum(2 * distribution / (distribution * len(distribution) + 1))
    expected_f1 = f1_scores.sum()
    return expected_f1

def f1_score_majority(distribution):
    d_maj = distribution.at[distribution['train'].idxmax(), 'test']
    f1 = 1 / len(distribution['test']) * (2 * d_maj / (1 + d_maj))
    return f1

class DatasetAnalytics(DatasetBuilder):
    def __init__(self):
        super().__init__()

    def get_sizes(self):
        sizes = dict()

        sizes["datasets"] = {}
        for dataset_name in self.datasets.keys():
            train, test, dev = self.datasets[dataset_name]()
            sizes["datasets"][dataset_name] = {
                "train": len(train),
                "test": len(test),
                "dev": len(dev),
                "full dataset": len(train)+len(test)+len(dev),
            }

        sizes["multilabel_datasets"] = {}
        for dataset_name in self.multilabel_datasets.keys():
            train, test, dev = self.multilabel_datasets[dataset_name]()
            sizes["multilabel_datasets"][dataset_name] = {
                "train": len(train),
                "test": len(test),
                "dev": len(dev),
                "full dataset": len(train)+len(test)+len(dev),
            }

        sizes["relation_datasets"] = {}
        for dataset_name in self.relation_datasets.keys():
            train, test, dev = self.relation_datasets[dataset_name]()
            sizes["relation_datasets"][dataset_name] = {
                "train": len(train),
                "test": len(test),
                "dev": len(dev),
                "full dataset": len(train)+len(test)+len(dev),
            }

        sizes["stance_datasets"] = {}
        for dataset_name in self.stance_datasets.keys():
            train, test, dev = self.stance_datasets[dataset_name]()
            sizes["stance_datasets"][dataset_name] = {
                "train": len(train),
                "test": len(test),
                "dev": len(dev),
                "full dataset": len(train)+len(test)+len(dev),
            }

        return sizes

    def data_description(self, train, test, dev):
        train_counts = train['label'].value_counts(normalize=True).rename("train")
        dev_counts = dev['label'].value_counts(normalize=True).rename("dev")
        test_counts = test['label'].value_counts(normalize=True).rename("test")

        distrib = pd.concat([train_counts, dev_counts, test_counts], axis=1)
        distrib.fillna(0, inplace=True)

        return distrib, f1_score_random(distrib), f1_score_majority(distrib)

class DebugDataset(DatasetBuilder):
    def __iter__(self):
        self.dataset_names = iter([
            'climate_sentiment',
            'climate_specificity',
            'sustainable_signals_review',
            'green_claims',
            'esgbert_action500',
        ])
        return self


class SortedNoRepeat(DatasetBuilder):
    def __iter__(self):
        self.dataset_names = ['esgbert_action500', 'green_claims', 'green_claims_3',
       'sustainable_signals_review', 'climate_specificity',
       'climate_sentiment', 'climate_commitments_actions',
       'climateFEVER_claim', 'climate_detection',
       'climate_tcfd_recommendations', 'esgbert_g', 'esgbert_s', 'esgbert_e',
       'esgbert_category_forest', 'esgbert_category_nature',
       'esgbert_category_biodiversity', 'esgbert_category_water',
       'environmental_claims', 'netzero_reduction', 'climateStance',
       'climateEng', 'sciDCC', 'ClimaINS', 'contrarian_claims', 'ClimaTOPIC',
       'lobbymap_pages', 'climatext', 'climateBUG_data']


        # dataset_to_exclude = [file[:-4] for file in os.listdir("experiment_results/cartography") if file.endswith('.tsv')]
        dataset_to_exclude = pd.read_csv(os.path.join(os.getcwd(), "experiment_results", "performances", "performances.csv"))['dataset_name'].unique().tolist()
        self.dataset_names = iter([name for name in self.dataset_names if name not in dataset_to_exclude])

        print("List of datasets:", self.dataset_names)

        return self
