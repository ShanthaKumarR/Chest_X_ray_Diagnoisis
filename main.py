import pandas as pd 
from data_analysis import Data_analysis
from image_preprocessing import image_preprocessing
import numpy as np
import seaborn as sn 
import matplotlib.pyplot as plt
from keras import backend as K
from densnet import Densenet_Model

train = pd.read_csv('train_A.csv')
test = pd.read_csv('test.csv')
validation = pd.read_csv('valid-small.csv')

X = Data_analysis(train, image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small')
y=X.data_insight()
X.data_leakage(validation)
image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small'
images = train['Image'].values
images = np.random.choice(images)
original_example = plt.imread(image_dir+'/'+images)

labels = list(train.keys())
labels.remove('Image')
labels.remove('PatientId')




def main():
    X_2 = image_preprocessing(original_example = original_example, image_dir = image_dir, data =train, labels = labels, batch_size = 1, target_w = 320, target_h = 320)
    y_2 = X_2.get_generator()
    #y_22 = X_2.Normalized_image(y_2)
    #X_2.Compare_image(y_22)

   

    values = np.mean(y_2.labels, axis=0)
    sn.barplot(values, labels, order=labels)
    plt.yticks(fontsize=13)
    plt.title("Frequency of Each Class", fontsize=14)
    plt.show()




    obj_1 = Densenet_Model(label = y_2.labels, epsilon = 1)
    positive_frequencies, negative_frequencies, w_p, w_n =obj_1.class_frequency_prediction()
    print(w_p, w_n)
    #loss_1 = obj_1.weighted_loss(y_predicted = y_pred_1, weight_pos = w_p, weight_neg=w_n)
    #loss_2 = obj_1.weighted_loss(y_predicted = y_pred_2, weight_pos = w_p, weight_neg=w_n)
    data = pd.DataFrame( {'Class': labels, "Positive_freq":positive_frequencies, "Negative_freq":negative_frequencies, "Total_freq" :  positive_frequencies+negative_frequencies}  )
    data.plot.bar(x="Class", y=["Positive_freq", "Negative_freq",  "Total_freq"], figsize=(15,15), color=['pink', 'gray', 'indigo']);
    plt.yticks(fontsize=16); plt.xticks(fontsize=16, rotation=20); plt.legend(fontsize =16);
    plt.show()

if __name__ == "__main__":
    main()