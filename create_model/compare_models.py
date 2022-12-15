from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from environs import Env

# Get environmental data
env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
PLOTS_PATH = env.str("PLOTS_PATH")
MODELS_PATH = env.str("MODELS_PATH")
SEED = env.str("SEED")

# Load train datasets
X_train_full = np.load(TRAIN_TEST_PATH + 'x_full.npy')
y_train_full = np.load(TRAIN_TEST_PATH + 'y_full.npy')

# Create dictionary to hold classifiers, it's names and the results of it's learning
comp_models = {
    'Decision_tree': [DecisionTreeClassifier()],
    'Random_forest': [RandomForestClassifier()],
    'MLP': [MLPClassifier()],
    'AdaBoost': [AdaBoostClassifier()],
    'Gauss': [GaussianNB()],
    'QDA': [QuadraticDiscriminantAnalysis()],
    'KNN': [KNeighborsClassifier()], 'NB': [GaussianNB()],
    'SVC_lin': [SVC()]
}

scales = [i for i in range(1000, 15001, 1000)]

df = pd.DataFrame()

# Train models and save its results
for i in scales:
    new_df = pd.DataFrame.from_dict(comp_models, orient='index')
    new_df.drop(columns=[0], inplace=True)
    new_df.reset_index(inplace=True)
    results_l, results_m = [], []
    for name, model in comp_models.items():
        kfold = model_selection.KFold(n_splits=8)
        results = model_selection.cross_val_score(model[0], X_train_full[:i], y_train_full[:i], cv=kfold,
                                                  scoring='accuracy')
        results_l.append(results)
        results_m.append(results.mean())
        print(f'{name} for {i} samples: {results.mean()} {results.std()}')

    new_df['values_list'], new_df['values_mean'], new_df['amount'] = results_l, results_m, [i for j in
                                                                                            range(len(results_l))]
    df = df.append(new_df)

# Build and save a box-plot for all classifiers by number of samples
for i in range(len(scales)):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Box-plot of the results of different models for {scales[i]} samples')
    ax = fig.add_subplot(111)
    plt.boxplot(df[df['amount'] == scales[i]].values_list.values)
    ax.set_xticklabels(comp_models.keys())
    ax.grid(linewidth=0.5)
    plt.savefig(PLOTS_PATH + f'comp/compare_models{scales[i]}.png', dpi=200)
    plt.show()

# Build and save a line-plot for all classifiers using number of samples as x-axis and score of model as y-axis
mean_by_classifier = df.groupby(['index', 'amount', 'values_mean']).size()
mean_by_classifier = mean_by_classifier.reset_index()
mean_by_classifier.drop(columns=[0], inplace=True)
mean_by_classifier.rename(columns={'index': 'classifier'}, inplace=True)
plt.figure(figsize=(15, 8))
sns.lineplot(x='amount',
             y='values_mean',
             hue='classifier',
             data=mean_by_classifier).set(title='Score by classifier by number of samples',
                                          xlabel='Number of samples',
                                          ylabel='Score')
plt.savefig(PLOTS_PATH + "compare_models_lineplot.png")
plt.show()
