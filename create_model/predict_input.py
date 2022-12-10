import pandas as pd
import numpy as np
import pickle
from environs import Env

# Get environmental data
env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
MODELS_PATH = env.str("MODELS_PATH")
PREPROCESSED_DATASET = env.str("PREPROCESSED_DATASET")

funding_total_usd_scaler = pickle.load(open(MODELS_PATH + 'funding_total_usd_scaler.pkl', 'rb'))
funding_rounds_scaler = pickle.load(open(MODELS_PATH + 'funding_rounds_scaler.pkl', 'rb'))
founded_at_scaler = pickle.load(open(MODELS_PATH + 'founded_at_scaler.pkl', 'rb'))

columns = ['funding_total_usd', 'funding_rounds', 'founded_at', 'region_Africa', 'region_Asia', 'region_Europe',
           'region_North America', 'region_Oceania', 'region_South America', 'category_list_0', 'category_list_1',
           'category_list_2', 'category_list_3']

# with open(MODELS_PATH + 'startup_success.json', 'r') as json_file:
#     loaded_model_json = json_file.read()
#     model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights(MODELS_PATH + "startup_success.h5")
model = pickle.load(open(MODELS_PATH + 'CART.sav', 'rb'))[0]


def predict_output(data):
    input_data = np.zeros(len(columns))

    region = data[0]
    input_data[3 + region] = 1

    funding = funding_total_usd_scaler.transform([[float(data[1]) * 1e6]])[0][0]
    input_data[0] = funding

    funding_rounds = funding_rounds_scaler.transform([[int(data[2])]])[0][0]
    input_data[1] = funding_rounds

    date_foundation = founded_at_scaler.transform([[pd.to_datetime(data[3], format='%Y-%m-%d', errors='coerce')
                                                  .value]])[0][0]
    input_data[2] = date_foundation

    input_data[9] = 1

    # return [round((i * 100), 2) for i in model.predict(input_data.reshape(1, -1))[0]]
    return model.predict_proba(input_data.reshape(1, -1))[0].argmax()


print(predict_output([1, '4', '2', '2004-08-07']))
