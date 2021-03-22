import numpy as np
import pandas as pd 
import seaborn as sn 
import matplotlib.pyplot as plt
import os
train_A = pd.read_csv('train_A.csv')
#print(train_A.columns)
#
#columns = train_A.sum()
#print(train_A.sum())
#for i in columns.keys():
    #print(i)

class Data_analysis:
    def __init__(self, data):
        self.data = data    
    def data_insight(self):
        print('The first five records are: \n ', self.data.head(5))
        print('The colum attributes(labels) are: \n', self.data.columns)
        print('The null value check: \n', self.data.info())
        print('No of unique patient Id is: \n', len(self.data['PatientId'].unique()))
        print(self.data.sum())
        return self.data.drop(['Image', 'PatientId'], 1, inplace=True)
    def image_visualization(self, image_dir):
        images = self.data['Image'].values
        images = [np.random.choice(images) for i in range(9)]
        plt.figure(figsize=(20,10))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            img = plt.imread(image_dir+'/'+images[i])
            plt.imshow(img, cmap= 'gray')
            plt.axis('off')
        plt.show()

        #print(images)

X = Data_analysis(train_A)
#y=X.data_insight()
X.image_visualization(image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small')