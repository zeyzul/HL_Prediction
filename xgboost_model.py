import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost

HUMAN_X = np.load('HUMAN_X.npy')
HUMAN_Y = np.load('HUMAN_Y.npy')
np.reshape(HUMAN_Y, 12929)  

# HUMAN_X_BERT = np.load('HUMAN_EMBEDDINGSX.npy')
# HUMAN_Y_BERT = np.load('HUMAN_EMBEDDINGSY.npy')
# np.reshape(HUMAN_Y_BERT, 12929)

# MOUSE_X = np.load('MOUSE_X.npy')
# MOUSE_Y = np.load('MOUSE_Y.npy')
# np.reshape(MOUSE_Y, 13717)

# MOUSE_X_BERT = np.load('MOUSE_EMBEDDINGSX.npy')
# MOUSE_Y_BERT = np.load('MOUSE_EMBEDDINGSY.npy')
# np.reshape(MOUSE_Y_BERT, 13717)

train_x, test_x, train_y, test_y = train_test_split(HUMAN_X, HUMAN_Y, test_size=0.4, random_state=42)

print('Training Human Features Shape:', train_x.shape)
print('Training Human Labels Shape:', train_y.shape)
print('Testing Human Features Shape:', test_x.shape)
print('Testing Human Labels Shape:', test_y.shape)

xgb = xgboost.XGBRegressor(eval_metric='rmsle', max_depth=None, learning_rate=0.01,min_child_weight=1, n_estimators=300)

# param_grid = {"max_depth":    [5, 10, 15, 20, None],
#               "n_estimators": [300, 400, 500, 600],
#               "min_child_weight": [ 1, 3, 5, 7 ],
#               "learning_rate": [0.01, 0.015, 0.02]}

# search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, cv=3, n_iter= 100, n_jobs=-1, verbose=2)
# search.fit(train_x, train_y)
# print("The best hyperparameters are:", search.best_params_)

print("Fitting the model...")
xgb.fit(train_x, train_y)

print("Predicting...")
predictions = xgb.predict(test_x)

test_r2_score = r2_score(test_y, predictions)
print("R2 score of predictions of test data:", test_r2_score)

mae_test = mean_absolute_error(test_y, predictions)
print(f'MAE test: {mae_test:.2f}')
