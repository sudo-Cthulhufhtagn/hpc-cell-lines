import tensorflow as tf
import numpy as np
import re
import cv2

neg = lambda x: x if x<0 else 0
relu = lambda x: x if x>0 else 0

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, 
                 df, 
                 batch_size,
                 input_size=(224, 224, 3),
                 channels=[6,7],
                 bbox_size=60,
                 n_classes=4,
                 avg_23=False,
                 shuffle=True):
        
        self.df = df.copy()
        self.bbox_size = bbox_size
        self.channels = channels
        self.n_classes = n_classes
        self.avg_23 = avg_23
        
        # self.classed_df = [
        #     len(df[df['class'] == i] ) for i in classes
        # ]
        # self.classed_df = [el / np.sum(self.classed_df) for el in self.classed_df]
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.__shuffle()
    
    def __shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.__shuffle()
    
    def __get_input(self, row):
        
        imgs = []
        for ch in self.channels:
            path = row['__URL']
            path = re.sub(r'-ch0\d', f'-ch0{ch}', path)
            imgs.append(cv2.imread(path, -1))
            
        image = np.stack(imgs, axis=-1)
            
        # xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        # image = tf.keras.preprocessing.image.load_img(path)
        # image_arr = tf.keras.preprocessing.image.img_to_array(image)

        # image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        # image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, label):
        return tf.keras.utils.to_categorical(label, num_classes=self.n_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_batch = [self.__get_input(row) for row in batches]
        y_batch = [self.__get_output(row) for row in batches['class']]

        return X_batch, y_batch
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)      
          
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size