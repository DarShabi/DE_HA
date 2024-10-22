DATA_PATH = '/Users/darshabi/Desktop/DE_HA/test_data_engineer.csv'
MAPPING_PATH= '/Users/darshabi/Desktop/DE_HA/test_data_engineer_aggregated_case_result.csv'
MISSING_VALUES_THRESHOLD = 25.0

# Multiplier used to determine the boundaries for outlier detection
OUTLIER_THRESHOLD_MULTIPLIER = 3.0

# Quantiles used for calculating the Interquartile Range (IQR)
LOWER_QUANTILE = 0.25
UPPER_QUANTILE = 0.75

# Modeling
TEST_SIZE_SPLIT = 0.2
RANDOM_STATE_SEED = 23
N_ESTIMATORS = 100