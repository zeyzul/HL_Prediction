import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# HUMAN_X = np.load('HUMAN_X.npy')
# HUMAN_Y = np.load('HUMAN_Y.npy')
# np.reshape(HUMAN_Y, 12928)  # ROW SIZE OF HUMAN DATA: 12929 / ROW SIZE OF MOUSE DATA: 13717 OR MANUALLY SELECTED

HUMAN_X_BERT = np.load('HUMAN_EMBEDDINGSX.npy')
HUMAN_Y_BERT = np.load('HUMAN_EMBEDDINGSY.npy')
np.reshape(HUMAN_Y_BERT, 12928)

# MOUSE_X = np.load('MOUSE_X.npy')
# MOUSE_Y = np.load('MOUSE_Y.npy')
# np.reshape(MOUSE_Y, 13717)

# MOUSE_X_BERT = np.load('MOUSE_EMBEDDINGSX.npy')
# MOUSE_Y_BERT = np.load('MOUSE_EMBEDDINGSY.npy')
# np.reshape(MOUSE_Y_BERT, 13717)


train_x, test_x, train_y, test_y = train_test_split(HUMAN_X_BERT, HUMAN_Y_BERT, test_size=0.25, random_state=42)

print('Training Human Features Shape:', train_x.shape)
print('Training Human Labels Shape:', train_y.shape)
print('Testing Human Features Shape:', test_x.shape)
print('Testing Human Labels Shape:', test_y.shape)

# LOAD THE MODE WITH PARAMETERS FROM HYPERTUNING
rf = RandomForestRegressor(n_estimators=100, min_samples_split=10, min_samples_leaf=2,
                           max_features="sqrt", max_depth=10, bootstrap=True, random_state=42)


# # HYPERTUNING
# n_estimators = [50, 100, 200, 300, 400]
# max_features = ['auto', 'sqrt']
# max_depth = [10, 20, 30, 40, 50, None]
# min_samples_split = [2, 5, 10, 20]
# min_samples_leaf = [1, 2, 4, 10]
# bootstrap = [True, False]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
#
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100,
#                               cv = 3, verbose=2, random_state=42, n_jobs = -1)

# rf_random.fit(train_x, train_y)
# print(rf_random.best_params_)


# TRAIN THE MODEL
print("Fitting the model...")
rf.fit(train_x, train_y)

# PREDICT
print("Predicting...")
predictions_test = rf.predict(test_x)

test_r2_score = r2_score(test_y, predictions_test)
print("R2 score of predictions of test data:", test_r2_score)

mae_test = mean_absolute_error(test_y, predictions_test)
print(f'MAE test: {mae_test:.2f}')


