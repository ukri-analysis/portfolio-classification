import json  # Standard library import
import re
import warnings

import numpy as np  # Third party imports
import pandas as pd
from bs4 import BeautifulSoup
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Suppress UserWarning to avoid cluttering output. Remove or modify as needed
warnings.filterwarnings("ignore", category=UserWarning)

# Maps special characters and HTML entities to replacements. While there
# is overlap with BeautifulSoup's HTML entity handling, LANG_MAP ensures
# correct replacements in non-HTML texts and encoded entity strings

LANG_MAP = {
    "\n": " ",          # Newline character to space
    "&quot;": '"',      # Double quotation mark
    "&amp;": "&",       # Ampersand
    "&lt;": "<",        # Less-than sign
    "&gt;": ">",        # Greater-than sign
    "&nbsp;": " ",      # Non-breaking space to space
    "&apos;": "'",      # Apostrophe
    "&cent;": "¢",      # Cent symbol
    "&pound;": "£",     # Pound sterling symbol
    "&yen;": "¥",       # Yen symbol
    "&euro;": "€",      # Euro symbol
    "&copy;": "©",      # Copyright symbol
    "&reg;": "®",       # Registered trademark symbol
    "&ndash;": "–",     # En dash
    "&mdash;": "—",     # Em dash
    "&lsquo;": "‘",     # Left single quotation mark
    "&rsquo;": "’",     # Right single quotation mark
    "&ldquo;": "“",     # Left double quotation mark
    "&rdquo;": "”",     # Right double quotation mark
    "&hellip;": "…",    # Ellipsis
    "&bull;": " ",      # HTML Bullet point to space
    "•": " ",           # Bullet point to space
    "*": " ",           # Asterisk to space
}

def clean_text(text):
    """
    Cleans the input text by removing HTML entities, replacing characters based on a predefined map,
    and normalizing whitespace and punctuation.

    Args:
    - text (str): The text to be cleaned.

    Returns:
    - str: The cleaned text.
    """

    # Decode HTML entities using LANG_MAP
    for entity, char in LANG_MAP.items():
        text = text.replace(entity, char)

    # Remove URLs
    text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text)

    # Now, use BeautifulSoup to remove any residual HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Condense and trim excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Removes spaces before punctuation marks (.,!?" at line ends or before spaces) for formatting.
    text = re.sub(r'\s+([?.!",](?:\s|$))', r'\1', text)

    return text


def load_data(filepath, analysis_column):
    """
    Loads data from CSV file, applies text cleaning on specified column, drops rows with empty values in column

    Args:
    - filepath (str): Path to the CSV file.
    - analysis_column (str): Name of the column containing text data to be cleaned.

    Returns:
    - pandas.DataFrame: The cleaned DataFrame with no empty values in the analysis column.
    """
    # Load the CSV file at the specified path into a DataFrame, using a try-except block to handle errors gracefully.
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {filepath} is empty or cannot be parsed as CSV.")

    if analysis_column not in df.columns:
        raise ValueError(f"The specified column '{analysis_column}' does not exist in the CSV file.")

    # Clean text in the analysis column and handle missing values
    df[analysis_column] = df[analysis_column].fillna('').apply(clean_text)

    # Drop rows with empty (after cleaning) analysis column values
    df = df[df[analysis_column].str.strip() != '']

    return df.reset_index(drop=True)


def save_metadata(metadata, filepath):
    """
    Writes given metadata to a specified JSON file.

    Args:
    - metadata (dict): The metadata to save.
    - filepath (str): Path to the target JSON file.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(metadata, f)
    except OSError as e:  # Using OSError as it's more commonly used in Python 3
        raise OSError(f"An error occurred while writing to {filepath}: {e.strerror}")


def chunk_text(text, tokenizer, max_length=512, overlap=50):
    """
    Splits the input text into overlapping segments according to specified length and overlap.

    Args:
    - text (str): The input text to be chunked.
    - tokenizer: The tokenizer object used to encode and decode text segments.
    - max_length (int): The maximum length of each text chunk.
    - overlap (int): The number of tokens to overlap between consecutive chunks. This overlap helps in
                     maintaining context across the chunks by repeating tokens at the end of one chunk
                     and the start of the next.

    Returns:
    - list of str: A list of text chunks that are potentially overlapping.
    """
    # Encode the text into a list of token ids
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    # Create overlapping chunks if text exceeds max_length
    if len(tokenized_text) > max_length:
        start = 0
        while start < len(tokenized_text):
            end = start + max_length
            chunk = tokenized_text[start:end]
            # Decode token ids back to text
            chunks.append(tokenizer.decode(chunk))
            start = end - overlap
    else:
        # If text is within the max_length, keep it as a single chunk
        chunks.append(text)

    return chunks


def encode_chunks_average(chunks, tokenizer, model, device):
    """
    Generates the average embedding vector for given text chunks using a specified tokenizer and model.

    Args:
    - chunks (list of str): Text chunks to encode.
    - tokenizer: Tokenizer object compatible with the model, used for encoding text chunks.
    - model: Model object used to generate embeddings for text chunks.
    - device: The computing device ('cpu' or 'cuda') where the model operations are performed.

    Returns:
    - torch.Tensor: The average embedding of the input text chunks as a tensor.
    """
    chunk_embeddings = []

    # Encode each chunk into embeddings
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1)  # Calculate the mean embedding for the chunk
        chunk_embeddings.append(chunk_embedding)

    # Calculate the average of all chunk embeddings
    if chunk_embeddings:
        embeddings_tensor = torch.stack(chunk_embeddings).mean(dim=0)  # Average across chunks
    else:
        embeddings_tensor = torch.zeros((1, model.config.hidden_size), device=device)  # Fallback to zero vector

    return embeddings_tensor


def create_embeddings(data_path, analysis_column, metadata_path, embeddings_path, device='cpu'):
    """
    Generates and saves embeddings for text data in a specified DataFrame column, along with DataFrame metadata.

    Args:
    - data_path (str): Path to the dataset file (including the file name and its extension, e.g., 'data.csv'),
                       expected to be loadable into a pandas DataFrame.
    - analysis_column (str): Name of the column containing text data for embedding generation.
    - metadata_path (str): File path (including the file name and .json extension, e.g., 'metadata.json')
                           where the DataFrame's metadata will be saved as JSON.
    - embeddings_path (str): Path where the embeddings tensor will be saved.
    - device (str): The device (CPU/GPU) to perform the computation on.

    Returns:
    None. The function saves the generated embeddings and metadata to the specified paths.
    """

    # Load dataset and extract the analysis column
    df = load_data(data_path, analysis_column)

    # Serialize DataFrame metadata as JSON
    metadata = df.to_dict(orient='index')

    # Use the save_metadata function to save the metadata to a file
    save_metadata(metadata, metadata_path)

    # Initialise tokenizer and model for embedding generation
    tokenizer = AutoTokenizer.from_pretrained('llmrails/ember-v1')
    model = AutoModel.from_pretrained('llmrails/ember-v1').to(device)

    # Prepare an empty tensor for storing embeddings
    embeddings = torch.empty((0, model.config.hidden_size), device=device)

    # Process each document to generate and collect embeddings
    for text in tqdm(df[analysis_column], desc="Creating embeddings"):
        # Split text into manageable chunks and encode to embeddings
        chunks = chunk_text(text, tokenizer, max_length=512, overlap=50)
        doc_embedding = encode_chunks_average(chunks, tokenizer, model, device)
        # Aggregate document embeddings
        embeddings = torch.cat((embeddings, doc_embedding), dim=0)

    # Save embeddings to file
    torch.save(embeddings, embeddings_path)