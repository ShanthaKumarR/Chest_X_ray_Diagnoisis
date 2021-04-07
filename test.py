import numpy as np 
import pandas as pd 

df = pd.DataFrame({'y_test': [1,1,0,0,0,0,0,0,0,1,1,1,1,1],
                   'preds_test': [0.8,0.7,0.4,0.3,0.2,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0],
                   'category': ['TP','TP','TN','TN','TN','FP','FP','FP','FP','FN','FN','FN','FN','FN']
                  })

y = df['y_test']
pred = df['preds_test']

def get_true_pos(y, pred, th=0.5):
    TP = 0
    thresholded_preds = pred >= th
    print(thresholded_preds)
    TP = np.sum((y == 1) & (thresholded_preds == 1))
    #print(TP)



get_true_pos(y = y, pred=pred, th=0.5)