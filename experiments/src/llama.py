import transformers
import torch

import sys
import linecache

from transformers import AutoTokenizer

import json
import os
import pandas as pd
from zero_shot import load_dict, extract_prompt, update_question, map_lobbymap_stance, prepare_content
import time


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))



def call_llama(history, pipeline):

    outputs = pipeline(
        history,
        max_new_tokens=2048,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        do_sample = False,
        temperature = 0.0
    )

    return outputs[0]["generated_text"][-1]["content"]


class ProgressLog:
    def __init__(self, total):
        self.total = total
        self.done = 0

    def increment(self):
        self.done = self.done + 1

    def __repr__(self):
        return f"Done runs {self.done}/{self.total}."


def call_llama_dataset(histories, pipeline):
    #TODO change according to this https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/56

    tokenized_histories = []
    for history in histories:
        prompt = pipeline.tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
        )
        tokenized_histories.append(prompt)


    outputs = pipeline(
        tokenized_histories,
        max_new_tokens=2048,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        do_sample = False,
        temperature = 0.0
    )
    return outputs





if __name__ == '__main__':
    path = "anonymized"
    model_id = "Meta-Llama-3.1-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(path+"/model_cache/"+model_id, padding_side="left")
    pipeline = transformers.pipeline(
        "text-generation",
        model=path+"/models_cache/"+model_id,
        model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": {"load_in_8bit": True}},
        tokenizer=tokenizer,
        device_map="cuda",
        batch_size=2
    )

    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]


    # Open the JSON file
    with open(os.path.join("mappings", "task_description.json"), 'r', encoding='utf-8') as file:
        task_descriptions = json.load(file)

    # Open the JSON file
    with open(os.path.join("mappings", "label_annotation.json"), 'r', encoding='utf-8') as file:
        label_readable_mapping = json.load(file)

    prompts = load_dict("prompts.json")

    dataset_to_run = [
                      "lobbymap_query",
                      "netzero_reduction",
                      "climate_specificity",
                      "climate_sentiment",
                      "climate_commitments_actions",
                      "climate_detection",
                      "climate_tcfd_recommendations",
                      "climatext",
                      "environmental_claims",
                      "ClimaTOPIC",
                      "climateFEVER_claim",
                      "climateBUG_data",
                      "lobbymap_pages",
                      "sustainable_signals_review",
                      "esgbert_e",
                      "esgbert_s",
                      "esgbert_g",
                      "esgbert_action500",
                      "esgbert_category_water",
                      "esgbert_category_forest",
                      "esgbert_category_biodiversity",
                      "esgbert_category_nature",
                      "sciDCC",
                      "green_claims",
                      "green_claims_3",
                      "contrarian_claims",
                      "climateStance",
                      "climateEng",
                      "ClimaINS_ours",
                      "gw_stance_detection",
                      "lobbymap_stance",
                      "climateFEVER_evidence",
                      "climaQA",
                      "logicClimate"]

    start_time = time.time()
    output = {}
    for dataset_name in dataset_to_run:
        print(dataset_name)
        test = pd.read_parquet(os.path.join("parquet", f"{dataset_name}.pkl"))
        if "cleaned_text" not in test.columns:
            test['clean_text'] = test['text'].copy()
        messages = []
        for index, row in test.iterrows():
            messages.append([
                {
                    "role": "system",
                    "content": "You are an AI annotator for NLP tasks related to climate-change. You will be provided with the description of a tasks. Please follow the instructions."
                },
                {
                    "role": "user",
                    "content": prepare_content(row, dataset_name, task_descriptions, prompts)
                }
            ])
        replies = call_llama_dataset(messages, pipeline)

        for i in range(len(replies)):
            index = replies[i][0]["generated_text"].rfind("end_header_id") + len("end_header_id") + 2
            reply = replies[i][0]["generated_text"][index:].replace("*", "")
            messages[i].append({"role": "assistant", "content": reply})
            print(messages[i])

        output[dataset_name] = messages

    wb = open(sys.argv[1], "w", encoding="utf-8")
    o_json = json.dumps(output)
    json.dump(o_json, wb)
    wb.close()

    print("--- %s seconds ---" % (time.time() - start_time))

