from keras import backend as K
import numpy as np


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
        def weighted_loss(y_true, y_pred):
                loss = 0.0
                for i in range(len(pos_weights)):
                        loss += K.mean(-(pos_weights[i] *y_true[:,i] * K.log(y_pred[:,i] + epsilon) 
                             + neg_weights[i]* (1 - y_true[:,i]) * K.log( 1 - y_pred[:,i] + epsilon)))

                return loss
        return weighted_loss




sess = K.get_session()
with sess.as_default() as sess:
    print("Test example:\n")
    y_true = K.constant(np.array([[1, 1, 1],[1, 1, 0],[0, 1, 0], [1, 0, 1]]))
    w_p = np.array([0.25, 0.25, 0.5])
    w_n = np.array([0.75, 0.75, 0.5])
    y_pred_1 = K.constant(0.7*np.ones(y_true.shape))
    loss = get_weighted_loss( pos_weights = w_p, neg_weights=w_n, epsilon=1)
    L1 = loss(y_true = y_true, y_pred = y_pred_1).eval()
    print(L1)

'''y_pred_1 = (0.7*np.ones(y_true.shape))
y_pred_2 = (0.3*np.ones(y_true.shape))
obj_1 = Densenet_Model(label = y_true, epsilon = 1)
loss = obj_1.weighted_loss(y_predicted = y_pred_1, weight_pos = w_p, weight_neg=w_n)
loss_2 = obj_1.weighted_loss(y_predicted = y_pred_2, weight_pos = w_p, weight_neg=w_n)'''
