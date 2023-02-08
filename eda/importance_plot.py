import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import pickle
import pandas as pd
from environs import Env

env = Env()
env.read_env()
PLOTS_PATH = env.str("PLOTS_PATH")
MODELS_PATH = env.str("MODELS_PATH")
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
PREPROCESSED_DATASET = env.str("PREPROCESSED_DATASET")

X_train = np.load(TRAIN_TEST_PATH + 'x_full.npy')
y_train = np.load(TRAIN_TEST_PATH + 'y_full.npy')


def plotly_scatterplots(model_importances, model_title):
    trace = go.Scatter(
        y=feature_dataframe[model_importances].values,
        x=feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=1,
            size=25,
            color=feature_dataframe[model_importances].values,
            colorscale='Portland',
            showscale=True
        ),
        text=feature_dataframe['features'].values
    )
    data = [trace]

    layout = go.Layout(
        autosize=True,
        title=model_title,
        hovermode='closest',
        yaxis=dict(
            title='Feature Importance',
            ticklen=5,
            gridwidth=2
        ),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='scatter2010')


# Create a dataframe with features
cols = pd.read_csv(PREPROCESSED_DATASET).columns.values

dec_tree_feature = pickle.load(open(MODELS_PATH + 'CART.sav', 'rb'))[0].feature_importances_
log_reg_feature = pickle.load(open(MODELS_PATH + 'LR.sav', 'rb'))[0].feature_importances_
nb_feature = pickle.load(open(MODELS_PATH + 'NB.sav', 'rb'))[0].feature_importances_

feature_dataframe = pd.DataFrame({'Features': cols,
                                  'Decision Tree': dec_tree_feature,
                                  'Logistic Regression': log_reg_feature,
                                  'Naive Bayes': nb_feature
                                  })
feature_dataframe = feature_dataframe[feature_dataframe.astype(bool).sum(axis=1) > feature_dataframe.shape[1] / 1.2]
model_importances = ['Decision Tree feature importances', 'Logistic Regression feature importances',
                     'Naive Bayes feature importances']

for importances, title in zip(model_importances, model_importances):
    plotly_scatterplots(importances, title)
