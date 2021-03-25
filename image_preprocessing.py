from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import seaborn as sn 
import matplotlib.pyplot as plt
import numpy as np 



train_A = pd.read_csv('train_A.csv')
image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small'


images = train_A['Image'].values
images = np.random.choice(images)
original_example = plt.imread(image_dir+'/'+images)

#Normalize images : new mean of the data will be zero, and the standard deviation of the data will be 1.
#In other words each pixel value in the image with a new value calculated by subtracting the mean and dividing by the standard deviation.
image_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization= True)
generator = image_generator.flow_from_dataframe(
        dataframe=train_A,
        directory=image_dir,
        x_col="Image", 
        y_col= ['Mass'], 
        class_mode="raw", 
        batch_size= 1, 
        shuffle=False,
        target_size=(320,320))


class image_preprocessing:
        def __init__(self, generator, original_example):
                self.generator = generator
                self.original_example=original_example

def Normalized_image(self):
        sn.set_style("white")
        generated_image, label = self.generator.__getitem__(0)
        plt.imshow(generated_image[0], cmap='gray')
        plt.colorbar()
        plt.title('Raw Chest X Ray Image')
        print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
        print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
        print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
        plt.show()
        
def Compare_image(self):              
        sn.set()
        plt.figure(figsize=(10, 7))
        sn.distplot(original_example.ravel(), 
                label=f'Original Image: mean {np.mean(self.original_example):.4f} - Standard Deviation {np.std(self.original_example):.4f} \n '
                f'Min pixel value {np.min(self.original_example):.4} - Max pixel value {np.max(self.original_example):.4}',
                color='blue', 
                kde=False)
        sn.distplot(self.generated_image[0].ravel(), 
                label=f'Generated Image: mean {np.mean(self.generated_image[0]):.4f} - Standard Deviation {np.std(self.generated_image[0]):.4f} \n'
                f'Min pixel value {np.min(self.generated_image[0]):.4} - Max pixel value {np.max(self.generated_image[0]):.4}', 
                color='red', 
                kde=False)
        plt.legend()
        plt.title('Distribution of Pixel Intensities in the Image')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('# Pixel')
        plt.show()

#X = image_preprocessing(generator, original_example)
#X.Compare_image()