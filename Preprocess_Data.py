import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from EDA import load_data, dataset_size
import Constants as c


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.
    Args:
    data (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Dataframe with duplicates removed.
    """
    return data.drop_duplicates()


def remove_columns_with_missing_values(data: pd.DataFrame,
                                       threshold: float = c.MISSING_VALUES_THRESHOLD) -> pd.DataFrame:
    """
    Remove columns from the dataframe that have a missing value percentage greater than a specified threshold.
    Args:
    data (pd.DataFrame): The input dataframe.
    threshold (float): The threshold for missing values percentage above which columns will be removed.

    Returns:
    pd.DataFrame: Dataframe with specified columns removed.
    """
    percent_missing = (data.isnull().sum() / len(data)) * 100
    columns_to_drop = percent_missing[percent_missing > threshold].index
    return data.drop(columns=columns_to_drop)


def impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the dataframe with the mean of their respective columns for numerical data,
    and with the mode for categorical data.
    Args:
    data (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Dataframe with missing values imputed.
    """
    # Impute numerical columns
    for column in data.select_dtypes(include=[np.number]).columns:
        if data[column].isnull().sum() > 0:
            data[column].fillna(data[column].mean(), inplace=True)

    # Impute categorical columns
    for column in data.select_dtypes(include=['object', 'category']).columns:
        if data[column].isnull().sum() > 0:
            most_frequent = data[column].mode()[0]  # Get the most frequent value
            data[column].fillna(most_frequent, inplace=True)

    return data


def handle_outliers(data: pd.DataFrame, threshold_multiplier: float = c.OUTLIER_THRESHOLD_MULTIPLIER) -> pd.DataFrame:
    """
    Handle outliers in the dataframe using a modified Interquartile Range (IQR) method.
    Args:
    data (pd.DataFrame): The input dataframe.
    threshold_multiplier (float): Multiplier used to determine the cut-off for outliers. Default is 3.0.

    Returns:
    pd.DataFrame: Dataframe with outliers handled gently.
    """
    for column in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[column].quantile(c.LOWER_QUANTILE)
        Q3 = data[column].quantile(c.UPPER_QUANTILE)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold_multiplier * IQR
        upper_bound = Q3 + threshold_multiplier * IQR

        # Handling outliers by capping and flooring
        data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
        data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])

    return data


def print_categorical_values(data: pd.DataFrame):
    """
    Print the names of categorical columns and their unique values.
    Args:
    data (pd.DataFrame): The input dataframe.
    """
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    print(f"There are {len(categorical_columns)} categorical columns.")
    for column in categorical_columns:
        print(f"\nColumn: {column}")
        print(f"Unique Values [{len(data[column].unique())}]: {data[column].unique()}")


def pre_correct_date(date_str):
    """Pre-correct known problematic date formats."""
    if date_str in ['19/02/2917', '03/02/2916']:  # Add more cases as identified
        return date_str.replace('291', '201')
    return date_str


def convert_dates(data):
    """Convert date columns and correct years that seem to be off by a century due to typos."""
    date_columns = ['a_enddate', 'moj_courtcaseopendate', 'moj_eventdate']
    current_year = pd.Timestamp.now().year

    for col in date_columns:
        # Pre-correct known problematic dates
        data[col] = data[col].apply(pre_correct_date)

        data[col] = pd.to_datetime(data[col], errors='coerce', dayfirst=True)
        data[col + '_year'] = data[col].dt.year
        data[col + '_month'] = data[col].dt.month
        data[col + '_day'] = data[col].dt.day

        # Output debug information
        # print(f"After extraction, {col + '_year'} contains NaN: {data[col + '_year'].isnull().any()}")

        # Reassemble the date
        data[col] = pd.to_datetime({
            'year': data[col + '_year'],
            'month': data[col + '_month'].fillna(1),
            'day': data[col + '_day'].fillna(1)
        }, errors='coerce')

        if data[col].isnull().any():
            print(f"Conversion issues still found in column {col}.")
            failed_conversions = data[data[col].isnull()][col]
            if not failed_conversions.empty:
                print(f"Failed to convert the following entries in column {col}:")
                print(failed_conversions)

    return data


def apply_label_encoding(data):
    """Apply label encoding to nominal categorical features."""
    categorical_columns = ['tnufa_businessunitidName', 'moj_courtidName', 'tnufa_bamacasetypeidName',
                           'tnufa_essencegroupidName', 'tnufa_claimessenceidName']
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data


def map_and_encode_target(data, mapping_df):
    """
    Map and encode the prediction target column using an external mapping DataFrame.
    Args:
    data (pd.DataFrame): The input dataframe.
    mapping_df (pd.DataFrame): DataFrame containing the mapping information.

    Returns:
    pd.DataFrame: Dataframe with the target column mapped and encoded.
    """
    mapping_df['tnufa_endreasonName'] = mapping_df['tnufa_endreasonName'].str.strip()
    mapping_df['result_type'] = mapping_df['result_type'].str.strip()
    mapping_dict = mapping_df.set_index('tnufa_endreasonName')['result_type'].to_dict()
    data['tnufa_endreasonName_mapped'] = data['tnufa_endreasonName'].map(mapping_dict)

    # Convert mapped categories to numerical labels
    label_encoder = LabelEncoder()
    data['tnufa_endreasonName_mapped'] = label_encoder.fit_transform(data['tnufa_endreasonName_mapped'])

    return data


def test_no_missing_values(data: pd.DataFrame):
    """
    Test to ensure there are no missing values in the dataframe.
    Args:
    data (pd.DataFrame): The input dataframe.
    """
    assert data.isnull().sum().sum() == 0, "There are still missing values in the dataframe."


def test_no_duplicates(data: pd.DataFrame):
    """
    Test to ensure there are no duplicate rows in the dataframe.
    Args:
    data (pd.DataFrame): The input dataframe.
    """
    assert data.duplicated().sum() == 0, "There are still duplicates in the dataframe."


def main():
    filepath = c.DATA_PATH
    mapping_filepath = c.MAPPING_PATH

    # Load main data and mapping data
    processed_data = load_data(filepath)
    mapping_data = load_data(mapping_filepath)

    # Complete initial preprocessing
    processed_data = remove_columns_with_missing_values(processed_data, threshold=c.MISSING_VALUES_THRESHOLD)
    processed_data = impute_missing_values(processed_data)
    processed_data = handle_outliers(processed_data, threshold_multiplier=c.OUTLIER_THRESHOLD_MULTIPLIER)
    processed_data = remove_duplicates(processed_data)

    # Proceed with data conversion
    processed_data = convert_dates(processed_data)
    processed_data = apply_label_encoding(processed_data)
    processed_data = map_and_encode_target(processed_data, mapping_data)
    print(processed_data['tnufa_endreasonName_mapped'].value_counts())

    processed_data.drop(columns=['tnufa_endreasonName'], inplace=True)

    # Run tests and output results
    test_no_missing_values(processed_data)
    test_no_duplicates(processed_data)
    print("All tests passed.")
    print("Preprocessing complete.")
    print(processed_data.head())
    print("Dataset Size after preprocessing:", dataset_size(processed_data))




if __name__ == "__main__":
    main()
