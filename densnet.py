import numpy as np
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import backend as K

class Densenet_Model:
    def __init__(self, label,  epsilon = 1e-7):
        self.epsilon = epsilon
        self.y_true = label


    def class_frequency_prediction(self):
        N = self.y_true.shape[0]
        #positive_frequencies = np.sum(labels, axis=0)/N
        #negative_frequencies = 1. - positive_frequencies
        #print(positive_frequencies)
        #print(negative_frequencies)
        positive_frequencies = np.sum(self.y_true == 0, axis=0)/N
        negative_frequencies = np.sum(self.y_true == 1, axis=0)/N
        weight_pos = negative_frequencies/N
        weight_neg = positive_frequencies/N
        return positive_frequencies, negative_frequencies, weight_pos, weight_neg
    
    def weighted_loss(self, y_predicted, weight_pos, weight_neg):
        loss = 0
        for i in range(len(weight_pos)):
            loss += -(np.mean((weight_pos[i] * self.y_true[:,i] * np.log(y_predicted[:,i] + self.epsilon)) + (weight_neg[i] * (1-self.y_true[:,i]) * np.log(1-y_predicted[:,i] + self.epsilon)),axis = 0))
        print(loss)
        return loss

    def pretrained_model1(self, weight_pos, weight_neg):
        base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
        x = GlobalAveragePooling2D()(base_model)
        predictions = Dense(len(labels), activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss=get_weighted_loss(weight_pos, weight_neg))




