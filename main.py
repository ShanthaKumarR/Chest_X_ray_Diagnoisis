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

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
validation = pd.read_csv('val.csv')

#sample image 
train_image_dir = 'D:/material_science/x-ray_data/images'
val_image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small'
images = train['Image'].values
images = np.random.choice(images)
original_example = plt.imread(train_image_dir+'/'+images)

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

def pretrained_model(labels, pos_weights, neg_weights):
    base_model = DenseNet121(include_top=False, weights="imagenet")
    x = base_model.output
    x_pool = GlobalAveragePooling2D()(x)
    predictions = Dense(len(labels), activation="sigmoid")(x_pool)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))
    print(model.summary())
    return model


def main():
    train_data = Data_analysis(train, image_dir = train_image_dir)
    train_data.data_leakage(validation)
    train_df =train_data.data_insight()
    label = train_df.drop(['Image'], 1, inplace=False).columns
    
    val_data = Data_analysis(validation, image_dir = val_image_dir)
    val_df =val_data.data_insight()

    test_data = Data_analysis(validation, image_dir = train_image_dir)
    test_df =test_data.data_insight()
    
    X_2 = image_preprocessing(original_example = original_example, image_dir = train_image_dir, train_df =train_df, 
    valid_df =val_df, test_df= test_df, labels = label,  batch_size = 10, val_dir = val_image_dir, test_dir=train_image_dir, target_w = 320, target_h = 320)
    train_generator= X_2.get_train_generator()
    y_true_train = train_generator.labels
    
    val_generator, test_generator = X_2.get_test_val_generator()    
    y_true_val = val_generator.labels
    y_true_test = test_generator.labels

    positive_frequencies, negative_frequencies, w_p, w_n = class_frequency_prediction(y_true_train)
      
    
    
    
    values = np.mean(y_true_train, axis=0)
    sn.barplot(values, label, order=label)
    plt.yticks(fontsize=13)
    plt.title("Frequency of Each Class", fontsize=14)
    plt.show()

    data = pd.DataFrame( {'Class': label, "Positive_freq":positive_frequencies, "Negative_freq":negative_frequencies, "Total_freq" :  positive_frequencies+negative_frequencies}  )
    data.plot.bar(x="Class", y=["Positive_freq", "Negative_freq",  "Total_freq"], figsize=(15,15), color=['Blue', 'Red', 'Yellow']);
    plt.yticks(fontsize=16); plt.xticks(fontsize=16, rotation=20); plt.legend(fontsize =16);
    plt.show( )

    model = pretrained_model(labels = label, pos_weights = w_p, neg_weights=w_n)
    history = model.fit_generator(train_generator, 
                              validation_data=val_generator,
                              steps_per_epoch= len(train_generator), 
                              validation_steps=len(val_generator), 
                              epochs = 30)
    #print(len(y_true_val), len(y_true_train), len(y_true_test))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save_weights("model.h5")
    predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))
    print(predicted_vals)

        
if __name__ == "__main__":
    main()
    




