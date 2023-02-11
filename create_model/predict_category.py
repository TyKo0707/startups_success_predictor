import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy import spatial
from environs import Env

env = Env()
env.read_env()
MFC = env.str("MFC")
GLOVE = env.str("GLOVE")

categories_df = pd.read_csv(MFC)
categories_df['Unnamed: 0'] = categories_df['Unnamed: 0'].apply(lambda x: x.lower().split(' ')[0])
categories_df.drop_duplicates(subset=['Unnamed: 0'], inplace=True)
categories_list = categories_df['Unnamed: 0'].values.tolist()
categories_df.set_index('Unnamed: 0', inplace=True)
data_desc = categories_df.describe()
first_quart, second_quart, third_quart = int(data_desc.loc['25%'].values[0]), int(data_desc.loc['50%'].values[0]), \
    int(data_desc.loc['75%'].values[0])

embeddings_dict = {}
with open(GLOVE, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for i in tqdm(range(len(lines))):
        values = lines[i].split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


def find_closest_embeddings_euc(name):
    embedding = embeddings_dict[name]
    return sorted(embeddings_dict.keys(),
                  key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


def find_correct_category(word):
    list_of_words = find_closest_embeddings_euc(word.lower())
    try:
        for i in list_of_words:
            if i in categories_list:
                if categories_df.loc[i].values[0] >= third_quart:
                    return 3
                elif first_quart <= categories_df.loc[i].values[0] < second_quart:
                    return 2
                else:
                    return 1
    except:
        return 0
