import pandas
import numpy as np
import pandas as pd
from environs import Env
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# Get environmental data
env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
PLOTS_PATH = env.str("PLOTS_PATH")
SEED = env.str("SEED")

# Load train datasets
X_train = np.load(TRAIN_TEST_PATH + 'x_full.npy')
y_train = np.load(TRAIN_TEST_PATH + 'y_full.npy')

columns = pd.read_csv('../preprocessed_dataset.csv').columns

# Create dictionary to hold classifiers, it's names and the results of it's learning
comp_models = {'CART': [DecisionTreeClassifier()], 'LR': [LogisticRegression()], 'LDA': [LinearDiscriminantAnalysis()],
               'KNN': [KNeighborsClassifier()], 'NB': [GaussianNB()]}

# Train and compare models using matplotlib (box-plot)
for name, model in comp_models.items():
    kfold = model_selection.KFold(n_splits=8)
    results = model_selection.cross_val_score(model[0], X_train, y_train, cv=kfold, scoring='accuracy')
    comp_models[name].append(results)
    print(f'{name}: {results.mean()} {results.std()}')

# Build and save a plot
fig = plt.figure()
fig.suptitle('Box-plot of the results of different models')
ax = fig.add_subplot(111)
plt.boxplot([i[1] for i in comp_models.values()])
ax.set_xticklabels(comp_models.keys())
ax.grid(linewidth=0.5)
plt.savefig(PLOTS_PATH + 'compare_models.png')
plt.show()
