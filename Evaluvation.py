import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from util import get_performance_metrics, get_true_pos, get_false_pos, get_false_neg, get_true_neg, get_curve
from sklearn.metrics import roc_auc_score, f1_score, roc_auc_score

train_results = pd.read_csv("train_preds.csv")
valid_results = pd.read_csv("valid_preds.csv")
#valid_results = pd.read_csv("valid_preds.csv")
class_labels = ['Cardiomegaly',
 'Emphysema',
 'Effusion',
 'Hernia',
 'Infiltration',
 'Mass',
 'Nodule',
 'Atelectasis',
 'Pneumothorax',
 'Pleural_Thickening',
 'Pneumonia',
 'Fibrosis',
 'Edema',
 'Consolidation']

pred_labels = [l + "_pred" for l in class_labels]
y = valid_results[class_labels].values
pred = valid_results[pred_labels].values


def val_class_freq():
    plt.xticks(rotation=90)
    plt.bar(x = class_labels, height= y.sum(axis=0));
    plt.show()


def get_accuracy(y, pred, th=0.5):
    TP = get_true_pos(y,pred,th)
    FP = get_false_pos(y,pred,th)
    TN = get_true_neg(y,pred,th)
    FN = get_false_neg(y,pred,th)
    return (TP + TN)/(TP+TN+FP+FN)


#Prevalence
def get_prevalence(y):
    prevalence = 0.0
    prevalence = np.mean(y)
    return prevalence
#Sensitivity and Specificity

def get_sensitivity(y, pred, th=0.5):
    sensitivity = 0.0
    TP = get_true_pos(y,pred,th)
    FN = get_true_neg(y,pred,th)
    sensitivity = TP/(TP+FN)
    return sensitivity

def get_specificity(y, pred, th=0.5):
    specificity = 0.0
    TN = get_true_neg(y,pred,th)
    FP = get_false_neg(y,pred,th)
    specificity = TN/(TN+FP)
    return specificity
#Positive predictive value (PPV) and Negative predictive value (NPV)
def get_ppv(y, pred, th=0.5):
    PPV = 0.0
    TP = get_true_pos(y,pred,th)
    FP = get_false_pos(y,pred,th)
    PPV = TP/(TP + FP)
    return PPV

def get_npv(y, pred, th=0.5):
    NPV = 0.0
    TN = get_true_neg(y,pred,th)
    FN = get_false_neg(y,pred,th)
    NPV = TN/(TN + FN)
    return NPV

#Confidence Intervals

# Calibration

result = get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score,f1=f1_score)


get_curve(y, pred, class_labels)
#Precision-Recall Curve
get_curve(y, pred, class_labels, curve='prc')