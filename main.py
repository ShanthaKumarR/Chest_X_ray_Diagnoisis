import pandas as pd 
from data_analysis import Data_analysis
from image_preprocessing import image_preprocessing
import numpy as np
import seaborn as sn 
import matplotlib.pyplot as plt

train = pd.read_csv('train_A.csv')
test = pd.read_csv('test.csv')
validation = pd.read_csv('valid-small.csv')

X = Data_analysis(train, image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small')
y=X.data_insight()
#classes = y.columns

X.data_leakage(validation)
#X.pixel_distrubution()
#X.image_visualization()
#X.class_imblance_predection(y)

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
    y_22 = X_2.Normalized_image(y_2)
    X_2.Compare_image(y_22)


    values = np.mean(y_2.labels, axis=0)
    sn.barplot(values, labels, order=labels)
    plt.yticks(fontsize=13)
    plt.title("Frequency of Each Class", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()