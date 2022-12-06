import pandas as pd
from keras.models import model_from_json
import numpy as np
import pickle
from environs import Env

# Get environmental data
env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
MODELS_PATH = env.str("MODELS_PATH")
PREPROCESSED_DATASET = env.str("PREPROCESSED_DATASET")

columns = list(pd.read_csv(PREPROCESSED_DATASET).columns.values)
columns.remove('status')


def input_data():
    funding_total_usd_scaler = pickle.load(open(MODELS_PATH + 'funding_total_usd_scaler.pkl', 'rb'))
    funding_rounds_scaler = pickle.load(open(MODELS_PATH + 'funding_rounds_scaler.pkl', 'rb'))
    founded_at_scaler = pickle.load(open(MODELS_PATH + 'founded_at_scaler.pkl', 'rb'))

    output_data = np.zeros(len(columns))

    # input the index of the region
    region_ind = int(input(
        "Input number with your region: \n1. Africa\n2. Asia\n3. Europe\n4. North America\n5. Oceania\n6. South America"))
    output_data[3 + region_ind] = 1

    funding = float(input('Input expected size of funding (in millions): ')) * 1e6
    funding = funding_total_usd_scaler.transform([[funding]])[0][0]
    print(funding)

    funding_rounds = int(input('Input expected size of funding rounds: '))
    funding_rounds = funding_rounds_scaler.transform([[funding_rounds]])[0][0]
    print(funding_rounds)

    output_data[0] = funding
    output_data[1] = funding_rounds

    date_foundation = input('Input date of foundation of the company (in format Y-m-d): ')
    date_foundation = founded_at_scaler.transform([[pd.to_datetime(date_foundation, format='%Y-%m-%d', errors='coerce')\
                                                    .value]])[0][0]
    print(date_foundation)
    output_data[2] = date_foundation

    category = input('Input the category of the company: ')
    output_data[10] = 1

    return output_data.reshape(1, -1)


with open(MODELS_PATH + 'startup_success.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(MODELS_PATH + "startup_success.h5")

input_data = input_data()

prediction = model.predict(input_data)
print(prediction[0])
