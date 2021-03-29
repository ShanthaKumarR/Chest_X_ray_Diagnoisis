from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import seaborn as sn 
import matplotlib.pyplot as plt
import numpy as np 



class image_preprocessing:
        def __init__(self,  original_example, image_dir, data, labels, batch_size, target_w = 320, target_h = 320):
                self.original_example=original_example
                self.image_dir = image_dir
                self.data = data
                self.batch_size = batch_size
                self.target_w = target_w
                self.target_h = target_h
                self.labels = labels
        def get_generator(self):
                #y_col = self.y_col.remove('Image')
                #y_col = self.y_col.remove('PatientId')
                image_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization= True)
                generator = image_generator.flow_from_dataframe(dataframe=self.data, directory=self.image_dir, x_col="Image", y_col=self.labels, 
                class_mode="raw", 
                batch_size= self.batch_size, 
                shuffle=False,
                target_size=(self.target_w, self.target_h))
                print(generator.labels)
                return generator


        def Normalized_image(self, generator):
                sn.set_style("white")
                generated_image, label = generator.__getitem__(0)
                plt.imshow(generated_image[0], cmap='gray')
                plt.colorbar()
                plt.title('Raw Chest X Ray Image')
                print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
                print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
                print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
                plt.show()
                return generated_image
                
        def Compare_image(self, generated_image):              
                sn.set()
                plt.figure(figsize=(10, 7))
                sn.distplot(self.original_example.ravel(), 
                        label=f'Original Image: mean {np.mean(self.original_example):.4f} - Standard Deviation {np.std(self.original_example):.4f} \n '
                        f'Min pixel value {np.min(self.original_example):.4} - Max pixel value {np.max(self.original_example):.4}',
                        color='blue', 
                        kde=False)
                sn.distplot(generated_image[0].ravel(), 
                        label=f'Generated Image: mean {np.mean(generated_image[0]):.4f} - Standard Deviation {np.std(generated_image[0]):.4f} \n'
                        f'Min pixel value {np.min(generated_image[0]):.4} - Max pixel value {np.max(generated_image[0]):.4}', 
                        color='red', 
                        kde=False)
                plt.legend()
                plt.title('Distribution of Pixel Intensities in the Image')
                plt.xlabel('Pixel Intensity')
                plt.ylabel('# Pixel')
                plt.show()

