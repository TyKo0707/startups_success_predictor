import pandas as pd
import numpy as np
from random import choices
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def get_numerical_distribution(df, key_feature, key_feature_val, feature_to_distr, n_bins):
    distr = []

    df_selected = df[df[key_feature] == key_feature_val][feature_to_distr]
    df_closed_len = len(df_selected)

    # get quantiles of 5 and 95 percents to derobust the data
    low, high = df_selected.quantile(.05), df_selected.quantile(.95)
    lspace = np.linspace(low, high, n_bins)

    prev = 0
    for ran in lspace:
        curr_choice = df_selected[(prev < df_selected) & (df_selected <= ran)]
        distr.append(len(curr_choice) / df_closed_len)
        prev = ran

    return distr, lspace


def get_categorical_distribution(df, key_feature, key_feature_val, feature_to_distr):
    df_selected = df[df[key_feature] == key_feature_val][feature_to_distr]

    if isinstance(df_selected.values[0], str):
        freq = df_selected.values
        most_freq = pd.Series(freq).value_counts() / len(df_selected)
        return most_freq.values, most_freq.index.values

    elif isinstance(df_selected.values[0], list):
        freq = []
        for i in df_selected.values:
            for j in i:
                freq.append(j)
        most_freq = pd.Series(freq).value_counts() / len(df_selected)
        return most_freq.values, most_freq.index.values


if __name__ == '__main__':
    df = pd.read_csv('../main_dataset.csv')

    # Splitting our category_list column by '|' and save this list to column
    df.category_list = df.category_list.str.split('|')

    df.drop(['name', 'company_index'], axis=1, inplace=True)

    columns = ['category_list', 'funding_total_usd', 'funding_rounds', 'founded_at']


    def generate_data(df, n, status):
        cols = df.columns.values
        new_df = pd.DataFrame(columns=cols)
        cache_distr = {}
        weights_region, rang_region = get_categorical_distribution(df, 'status', status, 'region')

        for i in tqdm(range(n)):
            temp_region = choices(rang_region, weights_region)[0]
            temp_data = {'category_list': '', 'funding_total_usd': 0, 'status': status, 'region': temp_region,
                         'funding_rounds': 0, 'founded_at': ''}
            for column in columns:
                if f'{temp_region}_{column}' not in cache_distr:
                    if isinstance(temp_data[column], str):
                        weights_column, rang_column = get_categorical_distribution(df[df.region == temp_region],
                                                                                   'status', status, column)
                        temp_data[column] = choices(rang_column, weights_column)[0]

                    elif isinstance(temp_data[column], int) or isinstance(temp_data[column], float):
                        weights_column, rang_column = get_numerical_distribution(df[df.region == temp_region],
                                                                                 'status', status, column, 100)
                        temp_data[column] = int(choices(rang_column, weights_column)[0])

                    cache_distr[f'{temp_region}_{column}'] = [weights_column, rang_column]
                else:
                    weights_column, rang_column = cache_distr[f'{temp_region}_{column}'][0], \
                                                  cache_distr[f'{temp_region}_{column}'][1]
                    if isinstance(temp_data[column], str):
                        temp_data[column] = choices(rang_column, weights_column)[0]
                    elif isinstance(temp_data[column], int) or isinstance(temp_data[column], float):
                        temp_data[column] = int(choices(rang_column, weights_column)[0])
            new_df = new_df.append(temp_data, ignore_index=True)

        return new_df


    new_status_closed, new_status_ipo = generate_data(df, 60000, 0), generate_data(df, 60000, 2)
    new_data = generate_data(df, 30000, 1)
    new_data.category_list = ['|'.join([i]) for i in new_data.category_list.values]
    new_status_closed.category_list = ['|'.join([i]) for i in new_status_closed.category_list.values]
    new_status_ipo.category_list = ['|'.join([i]) for i in new_status_ipo.category_list.values]

    df.category_list = ['|'.join(i) for i in df.category_list.values]

    df = df.append(new_status_closed, ignore_index=True)
    df = df.append(new_status_ipo, ignore_index=True)
    df = df.append(new_data, ignore_index=True)
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv('../extended_main_dataset_1.csv', index=False)
