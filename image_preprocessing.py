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
        def plot(self, img, title):
                plt.imshow(img, cmap='gray')
                plt.colorbar()
                plt.title(title)
                plt.show()

        def get_generator(self):
                image_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization= True)
                generator = image_generator.flow_from_dataframe(dataframe=self.data, directory=self.image_dir, x_col="Image", y_col=self.labels, 
                class_mode="raw", 
                batch_size= self.batch_size, 
                shuffle=False,
                target_size=(self.target_w, self.target_h))
                return generator


        def Normalized_image(self, generator):
                generated_image, label = generator.__getitem__(0)
                self.plot(img = generated_image[0], title = 'Normalized Chest X Ray Image')
                print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
                print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
                print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
                return generated_image
                
        def Compare_image(self, generated_image):              
                self.plot(img =self.original_example, title = 'Raw Chest X Ray Image')
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
        
        def class_frequency_prediction(self, labels):
                N = labels.shape[0]
                #positive_frequencies = np.sum(labels, axis=0)/N
                #negative_frequencies = 1. - positive_frequencies
                #print(positive_frequencies)
                #print(negative_frequencies)
                positive_frequencies = np.sum(labels == 0, axis=0)/N
                negative_frequencies = np.sum(labels == 1, axis=0)/N

                return positive_frequencies, negative_frequencies




