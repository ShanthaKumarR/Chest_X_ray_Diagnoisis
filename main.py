import pandas as pd 
from data_analysis import Data_analysis
from image_preprocessing import image_preprocessing
import numpy as np
import seaborn as sn 
import matplotlib.pyplot as plt
import keras 
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from util import get_roc_curve, compute_gradcam
import efficientnet.keras as efn
import tensorflow as tf
import tensorflow.keras.layers as L


train = pd.read_csv('train-small.csv')
test = pd.read_csv('test.csv')
validation = pd.read_csv('valid-small.csv')


#sample image 
train_image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small'
val_image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small'
test_image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small'
images = train['Image'].values
images = np.random.choice(images)
original_example = plt.imread(train_image_dir+'/'+images)

def class_frequency_prediction(y_true):
        N = y_true.shape[0]
        positive_frequencies = np.sum(y_true, axis=0)/N
        negative_frequencies = 1. - positive_frequencies
        #print(positive_frequencies)
        #print(negative_frequencies)
        #positive_frequencies = np.sum(y_true == 0, axis=0)/N
        #negative_frequencies = np.sum(y_true == 1, axis=0)/N
        weight_pos = negative_frequencies
        weight_neg = positive_frequencies
        return positive_frequencies, negative_frequencies, weight_pos, weight_neg


def get_weighted_loss( pos_weights, neg_weights, epsilon=1e-7):
        def weighted_loss(y_true, y_pred):
                loss = 0
                for i in range(len(pos_weights)):
                        loss_pos = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
                        loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
                        loss += loss_pos + loss_neg
                        #loss += -(K.mean((pos_weights[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon)) + (neg_weights[i] * (1-y_true[:,i]) * K.log(1-y_pred[:,i] + epsilon)),axis = 0))
                #print(loss)
                return loss
        return weighted_loss





def build_lrfn(lr_start=0.000002, lr_max=0.00010, 
               lr_min=0, lr_rampup_epochs=8, 
               lr_sustain_epochs=0, lr_exp_decay=.8):

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

lrfn = build_lrfn()

lr_schedule = LearningRateScheduler(lrfn, verbose=True)
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)





def pretrained_model(labels, pos_weights, neg_weights):
    base_model = DenseNet121(include_top=False, weights="imagenet")
    '''model = Sequential([
        efn.EfficientNetB1(
            input_shape=(128,128, 3),
            weights='imagenet',
            include_top=False),
        GlobalAveragePooling2D(),
        Dense(1024, activation = 'relu'), 
        Dense(len(labels), activation='sigmoid')
    ])'''
    #base_model = efn.EfficientNetB1(weights='imagenet', include_top=False)  
    x = base_model.output
    x_pool = GlobalAveragePooling2D()(x)
    #dense_layer = Dense(1024, activation = 'relu')(x_pool)
    predictions = Dense(len(labels), activation="sigmoid")(x_pool)
    model = Model(inputs=base_model.input, outputs=predictions)
    #model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights),  metrics = ['accuracy'])
    model.compile(optimizer=keras.optimizers.Adam( learning_rate=1e-4, amsgrad=False),
      loss = get_weighted_loss(pos_weights, neg_weights),
      metrics = ['accuracy'])
    print(model.summary())
    return model


def main():
    train_data = Data_analysis(train, image_dir = train_image_dir)
    train_data.data_leakage(validation)
    train_df =train_data.data_insight()
    label = train_df.drop(['Image'], 1, inplace=False).columns
    
    val_data = Data_analysis(validation, image_dir = val_image_dir)
    val_df =val_data.data_insight()

    test_data = Data_analysis(test, image_dir = test_image_dir)
    test_df =test_data.data_insight()
    
    X_2 = image_preprocessing(original_example = original_example, image_dir = train_image_dir, train_df =train_df, 
    valid_df =val_df, test_df= test_df, labels = label,  batch_size = 10, val_dir = val_image_dir, test_dir=test_image_dir, target_w = 320, target_h = 320)
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
    plt.show()

    model = pretrained_model(labels = label, pos_weights = w_p, neg_weights=w_n)
    #model.load_weights('efficent_net_b1_trained_weights.h5')
    model.load_weights('D:/material_science/pretrained_model.h5')
    '''history = model.fit_generator(train_generator, 
                              validation_data=val_generator,
                              steps_per_epoch= 1, 
                              validation_steps=1, 
                              epochs = 20, callbacks=[lr_schedule, checkpoint, EarlyStopping])'''
    #print(len(y_true_val), len(y_true_train), len(y_true_test))
    # summarize history for loss
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig('plot.png')
    #plt.show()
    #model.save_weights("model.h5")

   
    predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))
    df = pd.DataFrame(data=predicted_vals)
    df.to_csv('predections.csv')
    auc_rocs = get_roc_curve(label, predicted_vals, test_generator)
    #labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]
    #compute_gradcam(model, '00008270_015.png', test_image_dir, test, labels, labels_to_show)
if __name__ == "__main__":
    main()
    




