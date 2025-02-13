import os
import nltk
nltk.download('punkt')

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText
from unstructured.staging.base import convert_to_dataframe

from nltk.tokenize import word_tokenize

import pandas as pd
from src.tasks import climate_related, topic_detection

def process_pdf_to_text(file_path, save_path="case_study/machine_readable_data", strategy="hi_res", extract_images_in_pdf=False):
    """
    Process a PDF file to extract text and store it in a parquet file

    This function takes a PDF file and extracts the text from it. It then filters out
    any text elements that are less than 4 words long. The output is stored in a parquet
    file with the same name as the PDF file but with a .parquet extension.

    Parameters
    ----------
    file_path : str
        The path to the PDF file to be processed
    save_path : str
        The path to the folder where the parquet file should be saved
        (default is "case_study/machine_readable_data")
    strategy : str
        The strategy to use for partitioning the PDF (default is "hi_res")
    extract_images_in_pdf : bool
        Whether to attempt to extract images from the PDF (default is False)

    Returns
    -------
    None
    """
    elements = partition_pdf(filename=file_path, strategy=strategy, extract_images_in_pdf=extract_images_in_pdf)

    filtered_elements = [
        element for element in elements 
        if type(element)==NarrativeText
    ]

    df = convert_to_dataframe(filtered_elements)

    df = df[['page_number', 'filename', 'type', 'text']].copy()
    df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    df = df[df['word_count'] >= 4].copy()

    # create parquet file if not existant
    if not os.path.exists("machine_readable_data"):
        os.makedirs("machine_readable_data")
        
    df.to_parquet(f"{save_path}/{os.path.basename(file_path).replace('.pdf', '.parquet')}")
    print(f"Processed {os.path.basename(file_path)}")

def process_folder(path="case_study/data/", save_path="case_study/machine_readable_data"):
    """
    Process all PDF files in a given folder and save the extracted text to parquet files

    This function takes a folder of PDF files and processes them using the `process_pdf_to_text`
    function. The output is saved in a parquet file with the same name as the PDF file but with a
    .parquet extension.

    Parameters
    ----------
    path : str
        The path to the folder of PDF files to be processed
        (default is "case_study/data/")
    save_path : str
        The path to the folder where the parquet files should be saved
        (default is "case_study/machine_readable_data")

    Returns
    -------
    None
    """
    for file in os.listdir(path):
        if file.replace('.pdf', '.parquet') in os.listdir(save_path):
            print(f"{file} already processed")
        else:
            print(f"Processing {file}")
            process_pdf_to_text(file, save_path)
            
def processed_files_iterator(save_path="case_study/machine_readable_data"):
    """
    Iterator over all the processed files in the given save_path directory.

    Parameters
    ----------
    save_path : str
        The path to the folder where the parquet files are saved
        (default is "case_study/machine_readable_data")

    Yields
    ------
    str
        The filename of each processed parquet file
    """
    for file in os.listdir(save_path):
        if file.endswith('.parquet'):
            yield file, pd.read_parquet(os.path.join(save_path, file))

