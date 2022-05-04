import pandas as pd
from environs import Env

env = Env()
env.read_env()
MAIN_DATASET_PATH = env.str("MAIN_DATASET_PATH")

df = pd.read_csv(MAIN_DATASET_PATH)

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
