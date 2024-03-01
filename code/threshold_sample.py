import pandas as pd  # Third party imports
from sklearn.linear_model import LogisticRegression


def create_stratified_samples(dataframe, num_strata=10, num_samples_per_stratum=3):
    """
    Creates stratified samples from a dataframe based on the 'distance' column and adds a 'relevance' column
    for user labeling.

    This function divides the 'distance' into quantiles for stratified sampling, adds a 'relevance' column for
    user labeling, and saves the samples to a CSV file.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame must include a 'distance' column.
    - num_strata (int): Number of strata to divide the data into for sampling. Default is 10.
    - num_samples_per_stratum (int): Number of samples to draw from each stratum. Default is 3.

    Returns:
    - pd.DataFrame: A DataFrame containing the stratified samples, shuffled and with a 'relevance' column.
    """
    # Check if the dataframe has the required 'distance' column
    if 'distance' not in dataframe.columns:
        raise ValueError("Dataframe must include a 'distance' column.")

    # Add a 'relevance' column for user to fill out
    dataframe['relevance'] = None  # Placeholder for user labeling

    # Create quantile-based strata for 'distance'
    dataframe['distance_interval'] = pd.qcut(dataframe['distance'], q=num_strata, labels=False, duplicates='drop')

    samples = pd.DataFrame()  # Initialize dataframe to hold samples

    # Stratified sampling within each distance interval
    for i in range(num_strata):
        sampled = dataframe[dataframe['distance_interval'] == i].sample(
            n=min(num_samples_per_stratum, len(dataframe[dataframe['distance_interval'] == i])), replace=False)
        samples = pd.concat([samples, sampled], ignore_index=True)

    # Clean up by dropping the interval column
    samples.drop(columns=['distance_interval'], inplace=True)
    # Shuffle samples for randomness
    shuffled_samples = samples.sample(frac=1).reset_index(drop=True)

    # Save the shuffled samples to a CSV file, including the 'relevance' column for labeling
    shuffled_samples.to_csv("sample_output.csv", index=False)
    return shuffled_samples


def calculate_threshold(csv_path):
    """
    Calculates a threshold for distinguishing between 'relevant' and 'not relevant' samples based on logistic regression.

    This function reads a CSV file where the 'relevance' column has been labeled by the user, performs logistic regression,
    and calculates a threshold distance that separates 'relevant' from 'not relevant' samples.

    Parameters:
    - csv_path (str): Path to the CSV file containing the labeled samples.

    Returns:
    - float: The calculated threshold distance, rounded to 3 decimal places.
    """
    # Read the labeled samples from CSV
    data = pd.read_csv(csv_path)

    # Check if the data contains the required 'distance' and 'relevance' columns
    if 'distance' not in data.columns or 'relevance' not in data.columns:
        raise ValueError("CSV file must include both 'distance' and 'relevance' columns.")

    # Map 'yes'/'no' in 'relevance' to binary 1/0
    data['relevance'] = data['relevance'].map({'yes': 1, 'no': 0}).astype(int)

    # Initialize logistic regression model
    model = LogisticRegression(penalty=None)
    # Fit model on labeled data
    model.fit(data[['distance']], data['relevance'].values.reshape(-1, 1))

    # Calculate threshold from model coefficients
    threshold_distance = (-model.intercept_[0]) / model.coef_[0][0]
    return threshold_distance


def refined_sampling(dataframe, threshold, num_strata=10, num_samples_per_stratum=3):
    """
    Performs a second round of stratified sampling from the original dataframe, focusing on samples within a specific range
    around a given threshold distance.

    This function filters the dataframe for distances within +/- 0.05 of the threshold before performing stratified sampling
    on the filtered data. The function divides the filtered distances into quantiles for stratified sampling,
    and the resulting samples are shuffled.

    Parameters:
    - dataframe (pd.DataFrame): The original DataFrame from which to sample. Must include a 'distance' column.
    - threshold (float): The distance threshold around which to focus the sampling. This value is used to filter the data
      to distances within +/- 0.05 of the threshold.
    - num_strata (int): Number of strata to divide the filtered data into for sampling. Default is 10.
    - num_samples_per_stratum (int): Number of samples to draw from each filtered stratum. Default is 3.

    Returns:
    - pd.DataFrame: A DataFrame containing the stratified samples from the filtered range, shuffled.
    """

    # Filter the dataframe for distances within the specified range around the threshold, allowing negative values
    threshold_min = threshold - 0.05
    threshold_max = threshold + 0.05
    filtered_df = dataframe[(dataframe['distance'] >= threshold_min) & (dataframe['distance'] <= threshold_max)]

    # Use the initial stratified sampling function on the filtered dataframe
    return create_stratified_samples(filtered_df, num_strata, num_samples_per_stratum)

