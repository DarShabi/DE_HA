import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import Constants as c


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a specified file path with support for Hebrew encoding.
    Args:
    filepath (str): The path to the dataset file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='windows-1255')  # common for Hebrew files


def missing_entries(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of missing entries for each column in the dataframe.
    Args:
    data (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: A dataframe with columns for total missing and percentage missing.
    """
    missing_count = data.isnull().sum()
    percent_missing = (missing_count / len(data)) * 100
    missing_df = pd.DataFrame({'total_missing': missing_count, 'percent_missing': percent_missing})
    return missing_df


def column_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the types of columns in the dataframe and classify them as categorical or numerical.
    Args:
    data (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: A dataframe with type information for each column.
    """
    dtype_df = data.dtypes.apply(lambda x: 'categorical' if pd.api.types.is_object_dtype(x) else 'numerical')
    return pd.DataFrame({
        'column_name': dtype_df.index,
        'type': dtype_df.values
    })


def unique_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the unique values and their percentage distribution for each column in the dataframe.
    Args:
    data (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: A dataframe listing unique values and their percentage for each column.
    """
    unique_df = pd.DataFrame({col: [(data[col].nunique(),
                                     data[col].value_counts(normalize=True).to_dict())] for col in data.columns})
    unique_df = unique_df.transpose().reset_index()
    unique_df.columns = ['column_name', 'unique_values (count, distribution)']
    return unique_df


def dataset_size(data: pd.DataFrame) -> tuple:
    """
    Return the size of the dataset in terms of rows and columns.
    Args:
    data (pd.DataFrame): The input dataframe.

    Returns:
    tuple: Number of rows and columns in the dataset.
    """
    return data.shape


def count_duplicate_rows(data: pd.DataFrame) -> int:
    """
    Count the number of duplicate rows in the dataframe.
    Args:
    data (pd.DataFrame): The input dataframe.

    Returns:
    int: Number of duplicate rows in the dataset.
    """
    return data.duplicated().sum()


def plot_missing_values(data: pd.DataFrame) -> None:
    """
    Plot the percentage of missing values for each column in the dataframe with each bar colored differently.
    Args:
    data (pd.DataFrame): The input dataframe.

    Returns:
    None: Displays a bar chart showing the percentage of missing values per column.
    """
    percent_missing = (data.isnull().sum() / len(data)) * 100
    percent_missing.sort_values(inplace=True)
    plt.figure(figsize=(30, 10))
    colors = viridis(np.linspace(0, 1, len(percent_missing)))
    ax = percent_missing.plot(kind='bar', color=colors)
    plt.title('Percentage of Missing Values Per Column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage Missing (%)')
    plt.xticks(rotation=90)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=5)
    plt.tight_layout()
    plt.show()


def main():
    """
    Load data, perform EDA, and visualize results including the percentage of missing values for all columns.

    Returns:
    None: Outputs results of EDA.
    """
    filepath = c.DATA_PATH
    data = load_data(filepath)
    print("Dataset Size:", dataset_size(data))
    print("Number of Duplicate Rows:", count_duplicate_rows(data))
    print("Missing Entries:\n", missing_entries(data))
    print("Column Types:\n", column_types(data))
    print("Unique Values:\n", unique_values(data))
    plot_missing_values(data)


if __name__ == "__main__":
    main()
