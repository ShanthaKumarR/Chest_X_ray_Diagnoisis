from keras import backend as K
import numpy as np


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
        def weighted_loss(y_true, y_pred):
                loss = 0.0
                for i in range(len(pos_weights)):
                        loss += K.mean(-(pos_weights[i] *y_true[:,i] * K.log(y_pred[:,i] + epsilon) 
                             + neg_weights[i]* (1 - y_true[:,i]) * K.log( 1 - y_pred[:,i] + epsilon)))

                return loss
        return weighted_loss




sess = K.get_session()
with sess.as_default() as sess:
    print("Test example:\n")
    y_true = K.constant(np.array([[1, 1, 1],[1, 1, 0],[0, 1, 0], [1, 0, 1]]))
    w_p = np.array([0.25, 0.25, 0.5])
    w_n = np.array([0.75, 0.75, 0.5])
    y_pred_1 = K.constant(0.7*np.ones(y_true.shape))
    loss = get_weighted_loss( pos_weights = w_p, neg_weights=w_n, epsilon=1)
    L1 = loss(y_true = y_true, y_pred = y_pred_1).eval()
    print(L1)

'''y_pred_1 = (0.7*np.ones(y_true.shape))
y_pred_2 = (0.3*np.ones(y_true.shape))
obj_1 = Densenet_Model(label = y_true, epsilon = 1)
loss = obj_1.weighted_loss(y_predicted = y_pred_1, weight_pos = w_p, weight_neg=w_n)
loss_2 = obj_1.weighted_loss(y_predicted = y_pred_2, weight_pos = w_p, weight_neg=w_n)'''


class Generators:
    """
    Train, validation and test generators
    """
    def __init__(self, train_df, test_df):
        self.batch_size=32
        self.img_size=(64,64)
        
        # Base train/validation generator
        _datagen = ImageDataGenerator(
            rescale=1./255.,
            validation_split=0.25,
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
            )
        # Train generator
        self.train_generator = _datagen.flow_from_dataframe(
            dataframe=train_df,
            directory="../input/train/",
            x_col="id",
            y_col="label",
            has_ext=False,
            subset="training",
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=self.img_size)
        print('Train generator created')
        # Validation generator
        self.val_generator = _datagen.flow_from_dataframe(
            dataframe=train_df,
            directory="../input/train/",
            x_col="id",
            y_col="label",
            has_ext=False,
            subset="validation",
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=self.img_size)    
        print('Validation generator created')
        # Test generator
        _test_datagen=ImageDataGenerator(rescale=1./255.)
        self.test_generator = _test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory="../input/train/",
            x_col="id",
            y_col='label',
            has_ext=False,
            class_mode="categorical",
            batch_size=self.batch_size,
            seed=42,
            shuffle=False,
            target_size=self.img_size)     
        print('Test generator created')

        
# Create generators        
generators = Generators(train_df, test_df)
print("Generators created")