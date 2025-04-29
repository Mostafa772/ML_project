########################################################
### Don't forget to change the path to the data file ###
########################################################
import numpy as np
import pandas as pd


# function to perform one hot encoding on a specific column
def one_hot_encode(column_data):
    # Step 1: find all unique values in the column
    unique_categories = sorted(set(column_data))
    category_mapping = {category: idx for idx, category in enumerate(unique_categories)}
    
    # Create empty matrix with proper dimensions
    num_samples = len(column_data)
    num_categories = len(unique_categories)
    one_hot_matrix = np.zeros((num_samples, num_categories))
    
    # Fill the matrix with one-hot vectors
    for i, item in enumerate(column_data):
        one_hot_matrix[i, category_mapping[item]] = 1
    
    return one_hot_matrix, category_mapping


def load_data(MONK_NUM=1,train=True):
    MONK_NUM = str(MONK_NUM)
    if train:
        data = pd.read_csv(f"../ML_project/data/Monk_{MONK_NUM}/monks-{MONK_NUM}.train",
                 names=[0, 1, 2, 3, 4, 5, 6, "index"], delimiter=" ")
    else:
        data = pd.read_csv(f"../ML_project/data/Monk_{MONK_NUM}/monks-{MONK_NUM}.test",
                            names=[0, 1, 2, 3, 4, 5, 6, "index"], delimiter=" ")
    data.set_index("index", inplace=True)
    # test_data.head()
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    
    encoded_columns = {}
    for col in X:
        one_hot_encoded, category_to_index_map = one_hot_encode(data[col])
        encoded_columns[col] = pd.DataFrame(
            one_hot_encoded,
            columns=[f"{col}_{val}" for val in category_to_index_map.keys()],
        )

    # Combine one-hot encoded data with the target column
    one_hot_encoded_data = pd.concat(
        [encoded_columns[col] for col in X],
        axis=1,
    )
    print("one hot encoded data: ", one_hot_encoded_data.shape)
    # we are returning the values of the sereis which will be the target
    return one_hot_encoded_data, y.values



# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# # Print the shape of the resulting datasets
# print("Training Features Shape:", X_train.shape)
# print("Validation Features Shape:", X_val.shape)
# print("Training Target Shape:", y_train.shape)
# print("Validation Target Shape:", y_val.shape)
