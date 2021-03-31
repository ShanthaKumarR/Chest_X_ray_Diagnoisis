import pandas as pd 
from data_analysis import Data_analysis
from image_preprocessing import image_preprocessing
import numpy as np
import seaborn as sn 
import matplotlib.pyplot as plt
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras import backend as K

train = pd.read_csv('train_A.csv')
test = pd.read_csv('test.csv')
validation = pd.read_csv('valid-small.csv')

#sample image 
image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small'
images = train['Image'].values
images = np.random.choice(images)
original_example = plt.imread(image_dir+'/'+images)

def class_frequency_prediction(y_true):
        N = y_true.shape[0]
        #positive_frequencies = np.sum(labels, axis=0)/N
        #negative_frequencies = 1. - positive_frequencies
        #print(positive_frequencies)
        #print(negative_frequencies)
        positive_frequencies = np.sum(y_true == 0, axis=0)/N
        negative_frequencies = np.sum(y_true == 1, axis=0)/N
        weight_pos = negative_frequencies/N
        weight_neg = positive_frequencies/N
        return positive_frequencies, negative_frequencies, weight_pos, weight_neg


def get_weighted_loss( pos_weights, neg_weights, epsilon=1e-7):
        def weighted_loss(y_true, y_pred):
                loss = 0
                for i in range(len(pos_weights)):
                        loss += -(K.mean((pos_weights[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon)) + (neg_weights[i] * (1-y_true[:,i]) * K.log(1-y_pred[:,i] + epsilon)),axis = 0))
                print(loss)
                return loss
        return weighted_loss

def pretrained_model(y_true, pos_weights, neg_weights):
    base_model = DenseNet121(include_top=False, weights="imagenet")
    x = base_model.output
    x_pool = GlobalAveragePooling2D()(x)
    predictions = Dense(len(y_true), activation="sigmoid")(x_pool)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))
    return model


def main():
    X_1 = Data_analysis(train, image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small')
    label=X_1.data_insight().columns
    X_2 = image_preprocessing(original_example = original_example, image_dir = image_dir, data =train, labels = label, batch_size = 1, target_w = 320, target_h = 320)
    generator= X_2.get_generator()
    y_true = generator.labels

    
    X_22 = image_preprocessing(original_example = original_example, image_dir = image_dir, data =validation, labels = label, batch_size = 1, target_w = 320, target_h = 320)
    val_generator= X_22.get_generator()

    positive_frequencies, negative_frequencies, w_p, w_n = class_frequency_prediction(y_true)
    model = pretrained_model(y_true = y_true, pos_weights = w_p, neg_weights=w_n)  
    
    
    #X_1.data_leakage(validation)
    
    values = np.mean(y_true, axis=0)
    sn.barplot(values, label, order=label)
    plt.yticks(fontsize=13)
    plt.title("Frequency of Each Class", fontsize=14)
    plt.show()

    data = pd.DataFrame( {'Class': label, "Positive_freq":positive_frequencies, "Negative_freq":negative_frequencies, "Total_freq" :  positive_frequencies+negative_frequencies}  )
    data.plot.bar(x="Class", y=["Positive_freq", "Negative_freq",  "Total_freq"], figsize=(15,15), color=['pink', 'gray', 'indigo']);
    plt.yticks(fontsize=16); plt.xticks(fontsize=16, rotation=20); plt.legend(fontsize =16);
    plt.show()



    
    history = model.fit_generator(generator, 
                                validation_data=val_generator,
                                steps_per_epoch=2, 
                                validation_steps=25, 
                                epochs = 10)

    plt.plot(history.history['loss'])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Training Loss Curve")
    plt.show()
 
if __name__ == "__main__":
    main()
    




