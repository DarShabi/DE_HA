# Data Engineering and Analysis (DE_HA)

This repository, **DE_HA**, is a structured data engineering and analysis project designed to process, analyze, and model structured data. The pipeline includes exploratory data analysis, data preprocessing, modeling, and deployment using Docker.

## Project Structure and Files

### Data Files
- **test_data_engineer.csv**  
  - The raw test dataset used for exploratory data analysis and preprocessing. This is the initial input for the data pipeline.
- **test_data_engineer_aggregated_case_result.csv**  
  - A CSV file with aggregated results generated from the EDA or preprocessing phases, providing insights and summaries that inform model training or evaluation.
- **processed_data.json**  
  - Contains processed and cleaned data output by `Preprocess_Data.py`, used in testing, and serving as sample data for the API.

### Configuration and Constants
- **Constants.py**  
  - Defines key project constants, such as paths, model parameters, and other configuration values used across the pipeline. This centralizes commonly used variables, promoting modularity and consistency.
- **logging_config.py**  
  - Configures logging for the project, defining log formats, levels, and outputs to ensure traceability and support debugging.

### Code and Scripts
- **EDA.py**  
  - Performs Exploratory Data Analysis (EDA) on the dataset, providing insights into the data’s structure, distributions, and missing values. It generates initial visualizations and descriptive statistics, which are critical for understanding the data and preparing it for modeling.
- **Preprocess_Data.py**  
  - Handles data preprocessing tasks such as cleaning, normalization, encoding categorical variables, and managing missing values. The processed data is saved as a JSON file for further use in modeling and server deployment.
- **Modeling.py**  
  - Defines and trains a machine learning model using the processed data. The pipeline currently implements a Random Forest model, with capabilities for hyperparameter tuning, feature selection, and performance evaluation.

### Model and Logs
- **model.pkl**  
  - The trained machine learning model, serialized and saved for use in deployment. This file is loaded by `server.py` to provide predictions.
- **app.log**  
  - A log file capturing runtime logs, including information, warnings, and errors, providing traceability and useful for debugging and monitoring.

### Deployment
- **server.py**  
  - Sets up a Flask server to serve the trained model as an API endpoint, enabling easy deployment and access to predictions in real-time.
- **Dockerfile**  
  - Contains instructions to create a Docker image for the project, ensuring an isolated environment with essential dependencies and code files, and configuring `server.py` for deployment.
- **requirements.txt**  
  - Lists all the Python dependencies required for the project. Install them using `pip install -r requirements.txt` to ensure the environment is compatible with the code.
