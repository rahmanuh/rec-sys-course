from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prettytable import PrettyTable

import numpy as np
import pandas as pd

# Load all outputs/results
rf_df = pd.read_csv('random_forest.csv')
nn_df = pd.read_csv('nearest_neighbor.csv')
nb_df = pd.read_csv('naive_bayes.csv')

all_y_true = rf_df['y_test'].explode()
rf_y_predict = rf_df['y_predictions'].explode()
nn_y_predict = nn_df['y_predictions'].explode()
nb_y_predict = nb_df['y_predictions'].explode()

# Evaluate Random Forest
rf_mae = mean_absolute_error(all_y_true, rf_y_predict)
rf_mse = mean_squared_error(all_y_true, rf_y_predict)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(all_y_true, rf_y_predict)

# Evaluate Nearest Neighbor
nn_mae = mean_absolute_error(all_y_true, nn_y_predict)
nn_mse = mean_squared_error(all_y_true, nn_y_predict)
nn_rmse = np.sqrt(nn_mse)
nn_r2 = r2_score(all_y_true, nn_y_predict)

# Evaluate Naive Bayes
nb_mae = mean_absolute_error(all_y_true, nb_y_predict)
nb_mse = mean_squared_error(all_y_true, nb_y_predict)
nb_rmse = np.sqrt(nb_mse)
nb_r2 = r2_score(all_y_true, nb_y_predict)

# Present the evaluation results
table = PrettyTable()

table.field_names = ["Models", "MAE", "MSE", "RMSE", "R2 Score"]
table.add_row(["Random Forest", rf_mae, rf_mse, rf_rmse, rf_r2])
table.add_row(["Nearest Neighbor", nn_mae, nn_mse, nn_rmse, nn_r2])
table.add_row(["Naive Bayes", nb_mae, nb_mse, nb_rmse, nb_r2])

with open('evaluation_table.txt', 'w') as f:
    f.write(table.get_string())

print(table)