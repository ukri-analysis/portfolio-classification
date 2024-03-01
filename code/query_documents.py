import json  # Standard library import
import time

import numpy as np  # Third party imports
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer


def load_embeddings(embeddings_path, device):
    """
    Loads document embeddings from a file, ensuring they are moved to the specified device.

    Args:
        embeddings_path (str): Path to the embeddings file.
        device (str): The device to load the embeddings on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Document embeddings, moved to the specified device.
    """
    try:
        # Attempt to load the tensor and explicitly map it to the specified device
        embeddings = torch.load(embeddings_path, map_location=torch.device(device))
    except FileNotFoundError:
        # Raise an error if the embeddings file does not exist at the specified path
        raise FileNotFoundError(f"The embeddings file {embeddings_path} was not found.")
    except IOError as e:
        # Handle other input/output errors, such as permission issues
        raise IOError(f"An error occurred while loading the embeddings from {embeddings_path}: {e.strerror}")
    except RuntimeError as e:
        # Catch errors related to invalid device specifications
        raise RuntimeError(f"Failed to map the embeddings to the specified device '{device}': {e}")

    return embeddings

def load_metadata(metadata_path):
    """
    Loads and returns metadata from a specified JSON file.

    Args:
    - metadata_path (str): The path to the JSON file containing metadata.

    Returns:
    - dict: The metadata loaded from the file.
    """
    try:
        with open(metadata_path, 'r') as f:  # Explicitly using 'r' for readability
            return json.load(f)
    except OSError as e:  # Catching OSError to handle issues like file not being found
        raise OSError(f"An error occurred while reading from {metadata_path}: {e.strerror}")


def rank_documents(query_embedding, doc_embeddings):
    """
    Ranks documents based on cosine similarity scores between a query embedding and document embeddings.

    Args:
    - query_embedding (numpy.array): The embedding vector of the query.
    - doc_embeddings (numpy.array): An array of embedding vectors for documents.

    Returns:
    - numpy.array: Indices of documents sorted by descending similarity to the query.
    - numpy.array: Corresponding cosine similarity scores sorted in descending order.
    """
    # Calculate cosine similarity scores
    cos_scores = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)

    # Get sorted document indices and scores in descending order
    sorted_indices = np.argsort(-cos_scores)[0]
    sorted_scores = np.sort(-cos_scores)[0] * -1

    return sorted_indices, sorted_scores


def query_model(query, metadata_path, embeddings_path, device='cpu'):
    """
    Searches and ranks documents relevant to a given query using pre-trained model embeddings.

    This function loads the necessary data and model components, converts the query into an embedding using the
    same model used for document embeddings, and ranks documents based on their cosine similarity to the query
    embedding. The rankings and selected metadata for each document are then returned.

    Args:
    - query (str): The text query for searching relevant documents.
    - metadata_path (str): Path to the JSON file containing document metadata.
    - embeddings_path (str): Path to the file storing pre-computed document embeddings.
    - device (str): Computing device ('cpu' or 'cuda') for model operations.

    Returns:
    - pandas.DataFrame: A DataFrame containing ranked documents by relevance, including metadata such as
      rank, reference, funder, title, training data, word count, potential length warnings, and similarity distance.
    """
    start_time = time.time()

    # Load metadata and document embeddings from their respective paths
    metadata = load_metadata(metadata_path)
    doc_embeddings = load_embeddings(embeddings_path, device).cpu().numpy()

    # Set up the tokenizer and model for embedding the query
    tokenizer = AutoTokenizer.from_pretrained('llmrails/ember-v1')
    model = AutoModel.from_pretrained('llmrails/ember-v1').to(device)

    # Convert the query text to an embedding vector
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        output = model(**inputs)
    query_embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()

    # Normalize the query and document embeddings for similarity comparison
    query_embedding_norm = normalize(query_embedding)
    doc_embeddings_norm = normalize(doc_embeddings)

    # Obtain the ranked order of documents based on their similarity to the query
    ranked_indices, distances = rank_documents(query_embedding_norm, doc_embeddings_norm)

    # Assemble the search results with the relevant metadata for each ranked document
    results = []
    for rank, idx in enumerate(ranked_indices):  # Iterate over the ranked document indices
        str_idx = str(idx)  # Ensure the index is a string for metadata lookup
        if str_idx not in metadata:
            continue  # Skip the loop iteration if the document's metadata is missing

        doc_metadata = metadata[str_idx]
        results.append({
            "rank": rank + 1,
            "reference": doc_metadata.get("award_reference", "N/A"),
            "funder": doc_metadata.get("owner", "N/A"),
            "title": doc_metadata.get("application_title", "N/A"),
            "training_data": doc_metadata.get("training_text", "N/A"),
            "word_count": len(doc_metadata.get("training_text", "").split()),
            "warning": "WARNING" if len(doc_metadata.get("training_text", "").split()) <= 20 else "",
            "distance": distances[rank]  # Access the similarity distance for the ranked document
        })

    # Transform the list of results into a DataFrame for easier handling and display
    df_results = pd.DataFrame(results)

    end_time = time.time()
    print(f"The function took {round((end_time - start_time), 2)} seconds to complete")

    return df_results
