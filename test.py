from densnet import Densenet_Model
import numpy as np

y_true = np.array(
        [[1, 1, 1],
         [1, 1, 0],
         [0, 1, 0],
         [1, 0, 1]])

w_p = np.array([0.25, 0.25, 0.5])
w_n = np.array([0.75, 0.75, 0.5])

y_pred_1 = (0.7*np.ones(y_true.shape))
y_pred_2 = (0.3*np.ones(y_true.shape))
obj_1 = Densenet_Model(label = y_true, epsilon = 1)
loss = obj_1.weighted_loss(y_predicted = y_pred_1, weight_pos = w_p, weight_neg=w_n)
loss_2 = obj_1.weighted_loss(y_predicted = y_pred_2, weight_pos = w_p, weight_neg=w_n)
