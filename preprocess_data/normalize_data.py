import pandas as pd
import numpy as np
from environs import Env
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
MAIN_DATASET_PATH = env.str("MAIN_DATASET_PATH")


def normalize(df, by_col):
    # Normalize the DF using std_scale
    scaler = StandardScaler()
    dates_scaled = scaler.fit_transform(df[by_col].values.reshape((df.shape[0], 1)))
    df[by_col] = dates_scaled

    return df


df = pd.read_csv(MAIN_DATASET_PATH)

# Dropping useless columns
df.drop("Unnamed: 0", axis=1, inplace=True)

# Splitting our category_list column by '|' and save this list to column
df.category_list = df.category_list.str.split('|')

# get most frequent categories in dataframe and calculate it's quartiles
cl = []
for i in df.category_list.values:
    for j in i:
        cl.append(j)
mkc = pd.Series(cl).value_counts()[:200]
data_desc = mkc.describe()
first_quart, second_quart, third_quart = int(data_desc['25%']), int(data_desc['50%']), int(data_desc['75%'])

# replace old values (list of strings) with categorical ordinal values
for index, row in df.iterrows():
    try:
        new_val = max([mkc[i] for i in row.category_list])
        if new_val >= third_quart:
            new_val = 3
        elif first_quart <= new_val < second_quart:
            new_val = 2
        else:
            new_val = 1
    except:
        new_val = 0
    df.loc[index, "category_list"] = new_val

df.drop(['name', 'company_index'], axis=1, inplace=True)

# Converting date to numeric type
df.founded_at = pd.to_datetime(df.founded_at, format='%Y-%m-%d', errors='coerce').view('int64')

# Normalizing (StandardScaler) three main columns
df = normalize(df, 'founded_at')
df = normalize(df, 'funding_rounds')
df = normalize(df, 'funding_total_usd')

df = pd.get_dummies(df, columns=['region', 'category_list'])

df.to_csv('../preprocessed_dataset.csv')

# Choosing data and answers from dataframe
X = df.iloc[:, df.columns != 'status'].values
y = df.iloc[:, df.columns == 'status'].values
np.save(TRAIN_TEST_PATH + 'x_full.npy', X)
np.save(TRAIN_TEST_PATH + 'y_full.npy', y)

# Creating and saving train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

np.save(TRAIN_TEST_PATH + 'x_train.npy', X_train)
np.save(TRAIN_TEST_PATH + 'x_test.npy', X_test)
np.save(TRAIN_TEST_PATH + 'y_train.npy', y_train)
np.save(TRAIN_TEST_PATH + 'y_test.npy', y_test)
