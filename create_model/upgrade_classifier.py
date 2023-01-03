from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from environs import Env
from sklearn.model_selection import GridSearchCV

# Get environmental data
env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
PLOTS_PATH = env.str("PLOTS_PATH")
MODELS_PATH = env.str("MODELS_PATH")
SEED = env.int("SEED")

# Load train datasets
X_train = np.load(TRAIN_TEST_PATH + 'x_train.npy')[:10000]
y_train = np.load(TRAIN_TEST_PATH + 'y_train.npy')[:10000]
X_test = np.load(TRAIN_TEST_PATH + 'x_test.npy')[:3000]
y_test = np.load(TRAIN_TEST_PATH + 'y_test.npy')[:3000]


def search(classifier, params, filename):
    grid_search_cv = GridSearchCV(classifier(random_state=SEED), params, verbose=2, cv=3)
    grid_search_cv.fit(X_train, y_train)

    print('Best score through Grid Search : %.3f' % grid_search_cv.best_score_)
    print('Best Parameters : ', grid_search_cv.best_params_)

    best_params = grid_search_cv.best_params_
    df = pd.DataFrame.from_dict(best_params, orient='index')
    df.to_csv(MODELS_PATH + f'{filename}.csv')
    return grid_search_cv.best_estimator_


# Optimize hyper-params for DecisionTreeClassifier
parameters_tree = {'criterion': ['gini', 'entropy', 'log_loss'],
                   'max_depth': [2, 4, 6, 8, 10, None],
                   'max_leaf_nodes': list(range(2, 100)) + [None],
                   'min_samples_split': [2, 3, 4]}
dec_tree = search(DecisionTreeClassifier, parameters_tree, 'decision_tree_params')

print(f'Train score : {cross_val_score(estimator=dec_tree, X=X_train, y=y_train, cv=5, n_jobs=4)}')
print(f'Test score : {cross_val_score(estimator=dec_tree, X=X_test, y=y_test, cv=5, n_jobs=4)}')

# Optimize hyper-params for RandomForestClassifier
parameters_forest = {'criterion': ['gini', 'entropy', 'log_loss'],
                     'n_estimators': [i for i in range(100, 301, 100)],
                     'max_depth': [2, 4, 6, 8, 10, None],
                     'max_leaf_nodes': list(range(5, 101, 5)) + [None],
                     'max_features': [None, 'sqrt', 'log2'],
                     'min_samples_split': [2, 3, 4]}
ran_forest = search(RandomForestClassifier, parameters_forest, 'random_forest_params')

print(f'Train score : {cross_val_score(estimator=ran_forest, X=X_train, y=y_train, cv=5, n_jobs=4)}')
print(f'Test score : {cross_val_score(estimator=ran_forest, X=X_test, y=y_test, cv=5, n_jobs=4)}')
