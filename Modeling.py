import sys
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from Preprocess_Data import preprocess
import Constants as c
import logging
from logging_config import setup_logging

setup_logging()


def balance_classes(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Balance classes in the dataset by upsampling minority classes to match the majority class size.

    Args:
    data (pd.DataFrame): The input dataframe containing the target column.
    target_col (str): The name of the target column.

    Returns:
    pd.DataFrame: A dataframe with balanced class distribution.
    """
    max_size = data[target_col].value_counts().max()
    lst = [data]
    for class_index, group in data.groupby(target_col):
        lst.append(group.sample(max_size-len(group), replace=True))
    data = pd.concat(lst)
    return data


def split_data(data: pd.DataFrame, target_col: str, test_size: float = c.TEST_SIZE_SPLIT, random_state: int = c.RANDOM_STATE_SEED) -> tuple:
    """
    Split the data into training, validation, and testing sets.

    Args:
    data (pd.DataFrame): The input dataframe.
    target_col (str): The name of the target column.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.

    Returns:
    tuple: Contains training, validation, and test data as three dataframes.
    """
    try:
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state,
                                                 stratify=data[target_col])
        train_data, val_data = train_test_split(train_data, test_size=test_size, random_state=random_state,
                                                stratify=train_data[target_col])
        return train_data, val_data, test_data
    except Exception as e:
        logging.error("Failed to split data", exc_info=True)
        raise


def train_random_forest(train_data: pd.DataFrame, target_col: str) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the training data.

    Args:
    train_data (pd.DataFrame): The training data.
    target_col (str): The target column name.

    Returns:
    RandomForestClassifier: The trained Random Forest model.
    """
    try:
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        model = RandomForestClassifier(n_estimators=c.N_ESTIMATORS, random_state=c.RANDOM_STATE_SEED)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Failed to train the model", exc_info=True)
        raise


def evaluate_model(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Evaluate the given model using accuracy and generate a classification report.

    Args:
    model (RandomForestClassifier): The trained model.
    X (pd.DataFrame): The features of the data to be evaluated.
    y (pd.Series): The actual labels of the data.

    Outputs:
    Logs the accuracy and classification report.
    """
    try:
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")
    except Exception as e:
        logging.error("Failed to evaluate the model", exc_info=True)
        raise


def main():
    try:
        # Load and preprocess the entire data
        filepath = c.DATA_PATH
        data = preprocess(filepath)
        logging.info(f"Dataset size after preprocessing: {len(data)}")

        # Split the data into training, validation, and test sets
        train_data, val_data, test_data = split_data(data, 'tnufa_endreasonName_mapped')
        logging.info(f"Train set size: {len(train_data)}")
        logging.info(f"Validation set size: {len(val_data)}")
        logging.info(f"Test set size: {len(test_data)}")

        # Train the model
        model = train_random_forest(train_data, 'tnufa_endreasonName_mapped')

        # Evaluate the model on validation and test sets
        X_val = val_data.drop(columns=['tnufa_endreasonName_mapped'])
        y_val = val_data['tnufa_endreasonName_mapped']
        logging.info("----------Validation Results----------")
        evaluate_model(model, X_val, y_val)

        X_test = test_data.drop(columns=['tnufa_endreasonName_mapped'])
        y_test = test_data['tnufa_endreasonName_mapped']
        logging.info("----------Test Results----------")
        evaluate_model(model, X_test, y_test)

    except Exception as e:
        logging.error("An error occurred during processing: %s", str(e), exc_info=True)
        # Exit gracefully with an error status code
        sys.exit(1)


if __name__ == "__main__":
    main()
