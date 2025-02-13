import pandas as pd
import os
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
import numpy as np
import torch

def bootstrap_confidence_interval(y_true, y_pred, num_bootstrap_samples=1000, confidence_level=0.95):
    """Compute the bootstrap confidence interval for the F1 score."""
    if isinstance(y_true, pd.Series):
        y_true = y_true.reset_index(drop=True).values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.reset_index(drop=True).values

    bootstrap_scores = []
    for _ in range(num_bootstrap_samples):
        indices = resample(range(len(y_true)), replace=True)

        # multilabel
        if isinstance(y_pred[0], (list, np.ndarray)):
            y_pred_subset = [list(y_pred[i]) for i in indices]
            y_true_subset = [list(y_true[i]) for i in indices]

            list_of_strings = [str(row) for row in y_true_subset]
            if len(set(list_of_strings)) > 1:  # Ensure that the resampled data is valid for F1 score calculation
                bootstrap_score = f1_score(y_true=y_true_subset, y_pred=y_pred_subset, average='macro',
                                           zero_division=0.0)
                bootstrap_scores.append(bootstrap_score)
        # single label
        else:
            if len(set(y_true[indices])) > 1:  # Ensure that the resampled data is valid for F1 score calculation
                bootstrap_score = f1_score(y_true[indices], y_pred[indices], average='macro', zero_division=0.0)
                bootstrap_scores.append(bootstrap_score)

    bootstrap_scores = np.array(bootstrap_scores)
    lower_bound = np.percentile(bootstrap_scores, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_scores, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound
class Logger(object):
    def __init__(self, log_filename, reset=False):
        self.seed = 42

        #Todo: make sure not have double .csv
        self.log_filename = log_filename + ".csv"

        self.performances = pd.DataFrame()
        if not reset:
            self.performances = self.load()

    def set_seed(self, seed):
        self.seed = seed

    def add_f1_score(self, pipe, X_test, y_test, dataset_name, model_type, n_labels):
        y_pred = pipe.predict(X_test)

        report = classification_report(y_true=y_test, y_pred=y_pred, zero_division=0.0, output_dict=True)
        print(report)
        f1_lower, f1_upper = bootstrap_confidence_interval(y_true=y_test, y_pred=y_pred)
        print(f1_lower, f1_upper)

        self.add_record(dataset_name, model_type, report, n_labels, "f1_score", f1_upper, f1_lower)

    def add_precomputed_f1_score(self, y_pred, y_test, dataset_name, model_type, n_labels):
        report = classification_report(y_true=y_test, y_pred=y_pred, zero_division=0.0, output_dict=True)
        print(report)
        f1_lower, f1_upper = bootstrap_confidence_interval(y_true=y_test, y_pred=y_pred)
        print(f1_lower, f1_upper)

        self.add_record(dataset_name, model_type, report, n_labels, "f1_score", f1_upper, f1_lower)

    def add_trainer_f1_score_multi(self, trainer, test_dataset, dataset_name, model_type, n_labels, n_epoch):
        output = trainer.predict(test_dataset)
        logits = torch.sigmoid(torch.tensor(output.predictions))
        y_pred = (logits > 0.5).int().numpy().tolist()
        y_test = output.label_ids

        test_dataset_to_save = test_dataset
        test_dataset_to_save = test_dataset_to_save.add_column("y_pred", y_pred)
        test_dataset_to_save.to_parquet("experiment_results/performances/y_pred/"+dataset_name+"_"+model_type+"_"+str(self.seed)+".pkl")

        report = classification_report(y_true=y_test, y_pred=y_pred, zero_division=0.0, output_dict=True)
        print(report)
        f1_lower, f1_upper = bootstrap_confidence_interval(y_true=y_test, y_pred=y_pred)
        print(f1_lower, f1_upper)

        self.add_record(dataset_name, model_type, report, n_labels, "f1_score", f1_upper, f1_lower, n_epoch)

    def add_trainer_f1_score(self, trainer, test_dataset, dataset_name, model_type, n_labels, n_epoch):
        output = trainer.predict(test_dataset)
        y_pred = output.predictions.argmax(axis=-1)
        y_test = output.label_ids

        test_dataset_to_save = test_dataset
        test_dataset_to_save = test_dataset_to_save.add_column("y_pred", y_pred)
        test_dataset_to_save.to_parquet("experiment_results/performances/y_pred/"+dataset_name+"_"+model_type+"_"+str(self.seed)+".pkl")

        report = classification_report(y_true=y_test, y_pred=y_pred, zero_division=0.0, output_dict=True)
        print(report)
        f1_lower, f1_upper = bootstrap_confidence_interval(y_true=y_test, y_pred=y_pred)
        print(f1_lower, f1_upper)

        self.add_record(dataset_name, model_type, report, n_labels, "f1_score", f1_upper, f1_lower, n_epoch)

    def add_record(self, dataset_name, model_type, report, n_labels, performance_type, f1_upper, f1_lower, n_epoch=None):
        if ('samples avg' in report.keys()) and ('accuracy' not in report.keys()):
            report['accuracy'] = report['samples avg']['f1-score']

        new_row = pd.DataFrame({
            'dataset_name': [dataset_name],
            'model_type': [model_type],
            'performance': [report['macro avg']['f1-score']],
            'performance_type': [performance_type],
            'n_labels': [n_labels],
            'seed': [self.seed],
            "f1_upper": [f1_upper],
            "f1_lower": [f1_lower],
            "n_epoch": [n_epoch],
            "precision": [report['macro avg']['precision']],
            "recall": [report['macro avg']['recall']],
            "weighted_f1": [report['weighted avg']['f1-score']],
            "accuracy": [report['accuracy']]
        })
        self.performances = pd.concat([self.performances, new_row], ignore_index=True)

    def save(self, path="experiment_results/performances/"):
        self.performances.to_csv(path+self.log_filename, index=False)

    def load(self, path="experiment_results/performances/"):
        file_path = path+self.log_filename

        if os.path.exists(file_path):
            return pd.read_csv(path + self.log_filename)
        else:
            return self.performances

    def get_already_trained_datasets(self):
        if self.performances.empty:
            return []
        if "seed" not in self.performances.columns:
            print("Warning: missing seed column")
            return []
        if "dataset_name" not in self.performances.columns:
            print("Warning: missing dataset_name column")
            return []

        return self.performances[self.performances["seed"] == self.seed]['dataset_name'].unique()


class DebugLogger(Logger):
    def __init__(self, reset=False):
        if reset:
            self.performances = pd.DataFrame(
                columns=['dataset_name', 'model_type', 'performance', 'performance_type', 'dataset_size'])
        else:
            self.performances = self.load()

    def save(self, path="experiment_results/performances/"):
        self.performances.to_csv(path+'performances_debug.csv', index=False)

    def load(self, path="experiment_results/performances/"):
        print("Loading Performance Logs")
        if os.path.exists(path+'performances_debug.csv'):
            return pd.read_csv(path+'performances_debug.csv')
        else:
            return pd.DataFrame(
                columns=['dataset_name', 'model_type', 'performance', 'performance_type', 'dataset_size'])