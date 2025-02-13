import ast

from src.builder import DatasetBuilder
from src.model.longformer import train_longformer, train_multi_longformer
from src.model.distilbert import train_distilRoBERTa, train_multi_distilRoBERTa
from src.logger import Logger
from src.model.baseline import train_baselines, train_baselines_query_onehot, train_baselines_multilabel, train_baselines_relation
import os

import argparse
import pandas as pd

import torch
import random
import numpy as np
import re

def set_seed(seed, lonformer):
    """
    Everything should be reproducible
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # these are just for deterministic behaviour
    torch.backends.cudnn.benchmark = False
    if lonformer:
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_dataset(dataset_name):
    # dataset_path = os.path.join(os.getcwd(), "data", "cleaned_datasets", dataset_name)

    dataset_path = os.path.join(os.getcwd(), "data", "green_nlp_tasks", dataset_name)
    train = pd.read_parquet(os.path.join(dataset_path, "train.pkl"))
    test = pd.read_parquet(os.path.join(dataset_path, "test.pkl"))
    dev = pd.read_parquet(os.path.join(dataset_path, "dev.pkl"))

    if dataset_name in ["logicClimate"]:
        train["label"] = train["label"].apply(ast.literal_eval)
        test["label"] = test["label"].apply(ast.literal_eval)
        dev["label"] = dev["label"].apply(ast.literal_eval)

    return train, test, dev


def clean_datasets(train, test, dev, dataset_name, dataset_builder):
    if dataset_name in ["lobbymap_pages", "lobbymap_query", "lobbymap_stance"]:
        # We found that examples with less than 40 tokens are all labelled as 0, and correspond to pages with number or titles, but not actual content. Also we limit to 3900 because it is a query task (so there is the length of the text and the query)
        train = dataset_builder.filter(train, min_token=40, max_token=4000)
        test = dataset_builder.filter(test, min_token=40, max_token=4000)
        dev = dataset_builder.filter(dev, min_token=40, max_token=4000)
    else:
        train = dataset_builder.filter(train)
        test = dataset_builder.filter(test)
        dev = dataset_builder.filter(dev)

    train.drop_duplicates(inplace=True)
    test.drop_duplicates(inplace=True)
    dev.drop_duplicates(inplace=True)

    return train, test, dev

def generate_args(dataset_builder, dataset_list, logger):
    args = dict()

    for dataset_name in dataset_builder.datasets.keys():
        args[dataset_name] = {
            "function": dataset_builder.datasets[dataset_name],
            "training_function": train_baselines,
            "stratify_on": 'label',
            "label_columns": 'label',
            "input_columns": 'text',
            "classification_type": "classification",
            "balanced": "balanced",
            "weighted_loss": False,
        }

    for dataset_name in dataset_builder.relation_datasets.keys():
        args[dataset_name] = {
            "function": dataset_builder.relation_datasets[dataset_name],
            "training_function": train_baselines_relation,
            "stratify_on": "query",
            "label_columns": 'label',
            "input_columns": ['text', 'query'],
            "classification_type": "relation/stance",
            "balanced": "random",
            "weighted_loss": False,
        }

    for dataset_name in dataset_builder.stance_datasets.keys():
        args[dataset_name] = {
            "function": dataset_builder.stance_datasets[dataset_name],
            "training_function": train_baselines_query_onehot,
            "stratify_on": "label",
            "label_columns": 'label',
            "input_columns": ['text', 'query'],
            "classification_type": "relation/stance",
            "balanced": "balanced",
            "weighted_loss": False,
        }

    for dataset_name in dataset_builder.multilabel_datasets.keys():
        args[dataset_name] = {
            "function": dataset_builder.multilabel_datasets[dataset_name],
            "training_function": train_baselines_multilabel,
            "stratify_on": None,
            "label_columns": 'label',
            "input_columns": 'text',
            "classification_type": "multilabel",
            "balanced": "balanced",
            "weighted_loss": False,
        }

    for dataset_name in ['contrarian_claims', 'ClimaTOPIC']:
        args[dataset_name]["balanced"] = "weighted"


    for dataset_name in ["contrarian_claims",
                         "climateStance",
                         "climateEng",
                         "climate_tcfd_recommendations",
                         "climateFEVER_evidence",
                         "climateFEVER_evidence_climabench",
                         "sciDCC",
                         "logicClimate"
                         ]:
        args[dataset_name]["weighted_loss"] = True

    # Keep a subset of datasets
    args = {key: args[key] for key in dataset_list if key in args}

    # Skip already trained datasets
    remove_datasets = logger.get_already_trained_datasets()
    args = {key: value for key, value in args.items() if key not in remove_datasets}

    print("Not done: ***************")
    print(args.keys())

    return args

def experiment_loop(dataset_builder, logger, seed, dataset_list, batch_size, accumulation_steps, dataset_max_size=10000, do_train_baselines=False, do_train_longformer=False, do_train_distilRoBERTa=False):
    args = generate_args(dataset_builder, dataset_list, logger)

    for dataset_name in args.keys():
        print("DATASET:", dataset_name)

        print(args[dataset_name])

        # train, test, dev = args[dataset_name]["function"]()
        train, test, dev = load_dataset(dataset_name) # deterministic (load data)
        # train, test, dev = clean_datasets(train, test, dev, dataset_name, dataset_builder) # deterministic (drop column, filter)
        #
        # # determinisitc
        # if isinstance(args[dataset_name]['input_columns'], list):
        #     train = train.drop_duplicates(subset = args[dataset_name]['input_columns'] + [args[dataset_name]['label_columns']])
        #     test = test.drop_duplicates(subset = args[dataset_name]['input_columns'] + [args[dataset_name]['label_columns']])
        #     dev = dev.drop_duplicates(subset = args[dataset_name]['input_columns'] + [args[dataset_name]['label_columns']])
        # else:
        #     train = train.drop_duplicates(subset=[args[dataset_name]['input_columns'], args[dataset_name]['label_columns']])
        #     test = test.drop_duplicates(subset=[args[dataset_name]['input_columns'], args[dataset_name]['label_columns']])
        #     dev = dev.drop_duplicates(subset=[args[dataset_name]['input_columns'], args[dataset_name]['label_columns']])
        #
        # # deterministic (equilibrate climateFEVER_evidence)
        # if dataset_name == "climateFEVER_evidence":
        #     print("Downsampling to have a balanced dataset")
        #
        #     sampled_data = train[~(train['label'] == "NOT_ENOUGH_INFO")].copy()
        #     sampled_data = pd.concat([sampled_data,
        #                               train[(train['label'] == "NOT_ENOUGH_INFO")].sample(1539, replace=False,
        #                                                                                   random_state=seed)])
        #
        #     train = sampled_data
        #
        #
        # if train.shape[0] > dataset_max_size:
        #     print(f"Dataset size ({train.shape[0]}) is larger than {dataset_max_size}: Truncating the dataset to {dataset_max_size}, with a balanced label distribution for training and evaluation datasets. (Test dataset remain the same)")
        #     train, dev = dataset_builder.truncate(train=train, dev=dev, max_size=dataset_max_size, balanced=args[dataset_name]["balanced"], stratify_on=args[dataset_name]["stratify_on"])
        #     #TODO: add an automatic check to see if the dataset is balanced ? use args ?

        X_train = train[args[dataset_name]["input_columns"]]#.head(50)
        y_train = train[args[dataset_name]['label_columns']]#.head(50)

        X_test = test[args[dataset_name]["input_columns"]]#.head(50)
        y_test = test[args[dataset_name]['label_columns']]#.head(50)

        X_dev = dev[args[dataset_name]["input_columns"]]#.head(50)
        y_dev = dev[args[dataset_name]['label_columns']]#.head(50)

        if do_train_baselines:
            print("Training Baselines:")
            args[dataset_name]["training_function"](
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                dataset_name=dataset_name,
                logger=logger,
                seed=seed
            )

        if do_train_longformer:
            print("Training Longformer:")
            func = train_longformer
            if args[dataset_name]["classification_type"]=="multilabel":
                print("multilabel")
                func = train_multi_longformer
            func(
                X_train=X_train,
                y_train=y_train,
                X_val=X_dev,
                y_val=y_dev,
                X_test=X_test,
                y_test=y_test,
                model_save_path="model_save/distilbert/" + dataset_name,
                logging_dir="model_save/distilbert/" + dataset_name,
                dataset_name=dataset_name,
                logger=logger,
                classification_type=args[dataset_name]['classification_type'],
                seed=seed,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                weighted_training=args[dataset_name]["weighted_loss"],
            )

        if do_train_distilRoBERTa:
            print("Training distilRoBERTa")
            func = train_distilRoBERTa
            if args[dataset_name]["classification_type"] == "multilabel":
                print("multilabel")
                func = train_multi_distilRoBERTa
            func(
                X_train=X_train,
                y_train=y_train,
                X_val=X_dev,
                y_val=y_dev,
                X_test=X_test,
                y_test=y_test,
                model_save_path="model_save/distilbert/" + dataset_name,
                logging_dir="model_save/distilbert/" + dataset_name,
                dataset_name=dataset_name,
                logger=logger,
                classification_type=args[dataset_name]['classification_type'],
                seed=seed,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                weighted_training=args[dataset_name]["weighted_loss"],
            )

        logger.save()

        print(f"* * * * * * * * * * * * * < END: {dataset_name} *")





if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    #     for i in range(torch.cuda.device_count()):
    #         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # else:
    #     print("CUDA is not available.")

    parser = argparse.ArgumentParser(description="Experiments for greenwashing automatic detection, intermediary tasks")

    # Define an argument that takes a list of integers
    parser.add_argument(
        "--log",
        type=str,
        help="The name of the output log file",
        required = True,
    )

    parser.add_argument(
        "--seed_list",
        nargs='+',  # Accepts one or more arguments
        type=int,
        help="List of seeds"
    )

    # Define an argument that takes a list of strings
    parser.add_argument(
        "--dataset_list",
        nargs='+',  # Accepts one or more arguments
        type=str,
        help="A list of strings"
    )

    parser.add_argument(
        "-b", "--baseline",
        action="store_true",
        help="Run the baseline experiments"
    )

    parser.add_argument(
        "-l", "--longformer",
        action="store_true",
        help="Run the longformer experiments"
    )

    parser.add_argument(
        "-d", "--distilRoBERTa",
        action="store_true",
        help="Run the distilRoBERTa experiments"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="The size of the batch used during training",
    )

    parser.add_argument(
        "--accumulation_steps",
        type=int,
        help="The gradient accumulation parameter during training",
    )

    args = parser.parse_args()

    if args.longformer:
        if args.batch_size is None:
            parser.error("--batch_size required when running lonformer experiments")
        if args.accumulation_steps is None:
            parser.error("--accumulation_steps required when running lonformer experiments")

    print(f"Running baseline experiments: {args.baseline}")
    print(f"Running longformer experiments: {args.longformer}")

    if args.seed_list == None:
        args.seed_list = [
            42,
            43
        ]

    print(f"Running seeds: {args.seed_list}")

    if args.dataset_list == None:
        args.dataset_list = [
            'climate_sentiment',
            'climate_specificity',
            'sustainable_signals_review',
            'green_claims',
            'esgbert_action500'
        ]

    print(f"Running datasets: {args.dataset_list}")

    set_seed(42, lonformer=args.longformer)

    logger = Logger(log_filename=args.log)
    dataset_builder = DatasetBuilder(seed=42)

    for seed in args.seed_list:
        print(f"************* Seed {seed} *************")
        logger.set_seed(seed)
        experiment_loop(
            dataset_builder=dataset_builder,
            logger=logger,
            seed=seed,
            dataset_list=args.dataset_list,
            do_train_baselines=args.baseline,
            do_train_longformer=args.longformer,
            do_train_distilRoBERTa=args.distilRoBERTa,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps

        )
        print(f"********* End of Seed {seed} **********")


    print("END OF TRAINING")






