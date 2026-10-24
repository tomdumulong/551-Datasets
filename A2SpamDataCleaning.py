# COMP 551 - APPLIED MACHINE LEARNING
# Assignment 2 Data preprocessing code
# Due date: October 26, 2026
# Group 30: Adam DESJARDINS, Tom DUMULONG, Kiana NGAI

import pandas as pd
import numpy as np
RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)


# STEP 1 OF 3: ADD FEATURES NAMES TO DATASET
features = "https://raw.githubusercontent.com/tomdumulong/551-Datasets/refs/heads/main/spambase_names.csv"
nameless_data = "https://raw.githubusercontent.com/tomdumulong/551-Datasets/refs/heads/main/spambase.csv"

# [0] selects the first column, to_list() converts into a list
features_names = pd.read_csv(features, header = None)[0].to_list()

# Assign columns names
dataset_with_features_names = pd.read_csv(nameless_data, header = None, names = features_names)

# Export as .csv (which was then uploaded on GitHub, and reused below as "dataset_551_A2")
dataset_with_features_names.to_csv("dataset_551_Assignment2.csv", index = False)


# STEP 2 OF 3: DATA PREPROCESSING
# 1. Initialize a first dataset
url = "https://raw.githubusercontent.com/tomdumulong/551-Datasets/refs/heads/main/dataset_551_Assignment2.csv"
dataset_551_A2 = pd.read_csv(url) # First dataset, uncleaned, unsplit. The features names were added

# Just having all the features names in a list so I can easily iterate over them when need be
features_names = "https://raw.githubusercontent.com/tomdumulong/551-Datasets/refs/heads/main/spambase_names.csv"
features_names = pd.read_csv(features_names, header = None)[0].to_list()
features_names.pop() # Remove target feature from the features names; if reference needed, use target vector.

# 2. Look for "None" or "NaN"
print(pd.isna(dataset_551_A2).any(axis = "index"))

# 2. Look if any value is not an integer or a float. All 57 features must have real or integer values,
#    and target is binary. If anything else besides a float, integer, positive float/integer is within
#    the dataset, we report it.
for feature in features_names:
    # Go over every instance of every column feature
    for value in dataset_551_A2[feature]:
        if type(value) not in [float, int] or value < 0:
            # Returns the dataset_551_A2 row in which one of problematic values was found
            print(dataset_551_A2[dataset_551_A2[feature]==value])

# According to authors of dataset, there should not be any missing value, and both isna() call
# and search for non-numerical/negative values above returned no match. So far, so good...

# 3. No one-hot encoding required:
# Design matrix (derived below) -> 57 continuous features
# Target vector (derived below) is binary -> target vector never one-hot encoded

# 4. Leave all possible outliers in place. Yes, this feels counterintuitive -> more explanations in the report.

# CONCLUSION FOR DATA PREPROCESSING:
# dataset_551_A2 is approved for assignment usage. 

# STEP 3 OF 3: SPLIT + NORMALIZATION
# We next need to split with a 5/95 ratio (5% train, 95% test), and normalize the data.
# Those steps were however put directly in the notebook for simplicity.
