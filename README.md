# Portfolio classification methodology



## Summary

This repository includes all scripts needed to run thematic classification methodology described in [this paper](https://www.ukri.org/publications/investment-portfolio-classification/) .



## User guide

This user guide assumes you're following this methodology.

The steps are:

1. Creation of unique digital signatures for all awards
2. Keyword list & Calculation of metric of similarity
3. Threshold determination



### Requirements

All scripts use Python 3.9
Required packages are described in [requirements.txt](filepath).



### Input data

Input data must be formatted as a CSV file with only one column called `description` containing the text to model.



### Creation of unique digital signatures for all awards

To create the unique digital signatures for all awards run `create_embeddings` function from the `code/create_embeddings.py` script. Note that this process is resource intensive and may take a long time if your resources are limited.

Arguments for the function are:

  - `data_path`: Path to your CSV dataset.
  - `analysis_column`: The name of the column containing the text data - quoted?.
  - `metadata_path`: Desired path for saving the metadata JSON file.
  - `embeddings_path`: Desired path for saving the generated embeddings.
  - `device`: Choose either a CPU or a GPU for speed. Use `device = torch.device("cuda")` if your system has CUDA-compatible NVIDIA GPU support. Otherwise, default to `device = torch.device("cpu")` for CPU processing. Ensure your `torch` installation supports CUDA if opting for GPU acceleration. Check [pytorch documentation](link) for detail.

Outputs:

  - A JSON file containing metadata from your dataset. Which provides a reference to your original data.
  - A PyTorch tensor file (`.pt`) with the generated embeddings.



### Keyword list & **Calculation of metric of similarity**

To query and rank the documents as described in step 2 and step 3 of the methodology, run the `query_model` function from the `code/query_documents.py` script. This process involves loading the pre-trained model and document metadata to evaluate similarity between the query and existing document embeddings.

Arguments for the function are:

  - `query`: An unquoted series of key terms, separated by commas which dddddtext query to search for relevant documents.
  - `metadata_path`: Path to your metadata JSON file, which includes information about the documents.
  - `embeddings_path`: Path to your pre-computed document embeddings file.
  - `device`: Choose either a CPU or a GPU for speed. Use `device = torch.device("cuda")` if your system has CUDA-compatible NVIDIA GPU support. Otherwise, default to `device = torch.device("cpu")` for CPU processing. Ensure your `torch` installation supports CUDA if opting for GPU acceleration. Check [pytorch documentation](https://pytorch.org/) for detail.


Outputs: The function returns a pandas `DataFrame` with the following columns:

  - `rank`: The ranking of documents based on relevance.
  - `reference`: A unique reference for the document.
  - `funder`: The document's funder information.
  - `title`: The title of the document.
  - `description`: The text content of the document used for training.
  - `word_count`: The count of words in the document's text.
  - `warning`: A warning if the document's text is below a certain word count threshold.
  - `distance`: The cosine similarity distance, indicating how similar the document is to the query.



### Threshold determination

Selecting an optimal cut-off point (threshold) for similarity scores, to distinguish between documents that are relevant to the theme and those that are not, is achieved through a four-step process, utilising functions from the `code/threshold_sample.py` script.

**Step 1: Preliminary Sampling for Threshold Estimation**

Begin the process with the `create_stratified_samples` function to estimate an initial threshold by sampling from the full corpus.

Arguments for the function are:

- `dataframe`: The DataFrame containing documents with 'distance' metrics.
- `num_strata`: Optional. Defaults to 10 for dividing the data into strata.
- `num_samples_per_stratum`: Optional. Defaults to 3 for the number of samples from each stratum.

Output:

- A DataFrame with stratified samples, marked for user relevance labelling, providing an initial gauge of the threshold.

**Step 2: Computing the Initial Threshold**

The `calculate_threshold` function calculates a threshold from the labelled samples in Step 1.

Arguments for the function are:

- `csv_path`: Path to the CSV file with labelled samples.

Output:

- The initial threshold distance, essential for refining the sampling process.

**Step 3: Refined Sampling Around the Initial Threshold**

With the `refined_sampling` function, refine the sample selection by focusing on documents near the initial threshold. This step targets borderline cases to improve relevance classification accuracy.

Arguments for the function include:

- `dataframe`: The original DataFrame with document 'distance' metrics.
- `threshold`: The determined threshold distance from Step 2.
- `num_strata` and `num_samples_per_stratum`: Optional parameters for the sampling granularity and size.

Output:

- A DataFrame with samples focused around the threshold

**Step 4: Final Threshold Calculation**

After refining the sample set, run the `calculate_threshold` function again on the newly labelled samples to determine the final threshold.

Arguments for the function remain the same as in Step 2, applied to the new set of labelled samples from the refined sampling process.

Output:

- The final threshold distance, used to distinguish between relevant and not relevant documents based on similarity metrics.



## Licence

Unless stated otherwise, the codebase is released under [the MIT License](https://github.com/ukri-analysis/portfolio-classification/tree/set-up-project?tab=License-1-ov-file). This covers both the codebase and any sample code in the documentation.

The documentation is [© Crown copyright](http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/) and available under the terms of the [Open Government 3.0](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) licence.



