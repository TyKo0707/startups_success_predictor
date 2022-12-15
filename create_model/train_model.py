import numpy as np
from environs import Env
from sklearn.tree import DecisionTreeClassifier
import pickle

# Get environmental data
env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")
MODELS_PATH = env.str("MODELS_PATH")
SEED = env.int("SEED")

X_train = np.load(TRAIN_TEST_PATH + 'x_train.npy')
y_train = np.load(TRAIN_TEST_PATH + 'y_train.npy')
X_test = np.load(TRAIN_TEST_PATH + 'x_test.npy')
y_test = np.load(TRAIN_TEST_PATH + 'y_test.npy')

# params = pd.read_csv(MODELS_PATH + 'decision_tree_params.csv')
classifier = DecisionTreeClassifier(criterion='entropy', min_samples_split=2)
classifier.fit(X_train, y_train)

print('Train score : %.3f' % classifier.score(X_train, y_train))
print('Test score : %.3f' % classifier.score(X_test, y_test))

pickle.dump(classifier, open(MODELS_PATH + 'decision_tree.sav', 'wb'))
