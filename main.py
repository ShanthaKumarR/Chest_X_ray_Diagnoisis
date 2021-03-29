import pandas as pd 
from data_analysis import Data_analysis

train = pd.read_csv('train_A.csv')
test = pd.read_csv('test.csv')
validation = pd.read_csv('valid-small.csv')

X = Data_analysis(train, image_dir = 'D:/material_science/rwa/AI-For-Medicine-Specialization-master/AI for Medical Diagnosis/Week 1/nih/images-small')
y=X.data_insight()
#X.pixel_distrubution()
#X.image_visualization()
X.class_imblance_predection(y)