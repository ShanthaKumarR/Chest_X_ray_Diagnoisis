import numpy as np 

y_true = np.array([[1], [1], [1], [0], [0]])
print(f"the y_true: \n {y_true}")

y_pred_1 = 0.9 * np.ones(y_true.shape)
print(f"the y_true: \n {y_pred_1}")
y_pred_2 = 0.1 * np.ones(y_true.shape)
print(f"the y_true: \n {y_pred_2}")