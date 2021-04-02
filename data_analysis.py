import numpy as np
import pandas as pd 
import seaborn as sn 
import matplotlib.pyplot as plt
import os



class Data_analysis:
    def __init__(self, data, image_dir):
        self.data = data  
        self.image_dir = image_dir  

    def data_insight(self):
        print('The first five records are: \n ', self.data.head(5))
        print('The colum attributes(labels) are: \n', self.data.columns)
        print('The null value check: \n', self.data.info())
        print('No of unique patient Id is: \n', len(self.data['PatientID'].unique()))
        print(self.data.sum())
        train = self.data.drop(['PatientID'], 1, inplace=False)
        #labels = self.data.drop(['Image'], 1, inplace=False).columns
        return train
    def image_visualization(self):
        images = self.data['Image'].values
        images = [np.random.choice(images) for i in range(9)]        
        plt.figure(figsize=(20,10))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            img = plt.imread(self.image_dir+'/'+images[i])
            plt.imshow(img, cmap= 'gray')
            plt.axis('off')
        plt.show()

    def pixel_distrubution(self):
        images = self.data['Image'].values
        images = np.random.choice(images)
        original_example = plt.imread(self.image_dir+'/'+images)
        print('The dimention of the image is:', original_example.shape)
        print('pixel mean: ', np.mean(original_example))
        print('pixel Standard Deviation: ', np.std(original_example))
        #plt.figure(figsize=(12,4))
        #plt.subplot(121)
        #plt.imshow(original_example, cmap= 'gray')
        #plt.show()        
        plt.figure(figsize=(12,4))
        plt.subplot(121)
        sn.distplot(original_example, kde=True)
        plt.xlabel('Pixel Intensity', fontsize=14)
        plt.ylabel('# Pixels in Image', fontsize=14)
        plt.subplot(122)
        sn.distplot(original_example, kde=False)
        plt.xlabel('Pixel Intensity', fontsize=14)
        plt.ylabel('Pixels in Image', fontsize=14)

    def class_imblance_predection(self, clean_data):
        print(clean_data.sum())
        sn.barplot(clean_data.sum().values, clean_data.sum().index, color='g')
        plt.title('Distribution of Classes for Training Dataset', fontsize=14)
        plt.xlabel('Number of Patients', fontsize=14)
        plt.ylabel('Diseases', fontsize=14)
        plt.show()

    def data_leakage(self, dataset_two):
        dataset_one_overlap = []
        dataset_two_ovelap = []
        dataset_one_ids = set(self.data.PatientID.values)
        dataset_two_ids = set(dataset_two.PatientID.values)
        overlap = list(dataset_one_ids.intersection(dataset_two_ids))
        if overlap != 0:
            print(f"There are {len(overlap)} datas are overlaping")
            print(overlap)
            for id in range(len(overlap)):
                dataset_one_overlap.extend(self.data.index[self.data['PatientID'] ==  overlap[id]].tolist())
                dataset_one_overlap.extend(dataset_two.index[dataset_two['PatientID'] ==  overlap[id]].tolist())
        return dataset_one_overlap, dataset_two_ovelap

    
    def bar_plot(self):
        plt.figure(figsize=(15,4))
        sn.barplot(x="Class", y="Value", hue="Label" ,data=data);
        plt.xticks(rotation=90);
        plt.yticks(fontsize=16); plt.xticks(fontsize=16); plt.legend(fontsize =16);




