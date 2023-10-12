import os
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
                 top_path,
                 input_size=(224, 224),
                 channels=[6,7],
                 bbox_size=60,
                 n_classes=4,
                 normalize_ch_color=False,
                 avg_23=False,
                 shuffle=True):
        
        self.df = df.copy()
        self.top = top_path
        self.bbox_size = bbox_size
        self.channels = channels
        self.n_classes = n_classes
        self.avg_23 = avg_23
        self.normalize_ch_color = normalize_ch_color
        
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
            path = os.path.join(self.top, path)
            img = cv2.imread(path, -1)
            
            keypoints = row['keypoints']
            x, y = keypoints[0], keypoints[1]
            d = self.bbox_size
            max_x = img.shape[1]
            max_y = img.shape[0]
            start = (
                relu(x - d) - relu(x+d-max_x), 
                min(y + d, max_y) + relu(d-y)
            )
            stop = (
                min(x+d, max_x) + relu(d-x),
                relu(y-d) - relu(y+d-max_y)
            )
            
            imgs.append(
                cv2.resize(img[stop[1]:start[1], start[0]:stop[0]], self.input_size[:2])
                )
            
        image = np.stack(imgs, axis=-1)
        
        if self.normalize_ch_color:
            # using mean and standard deviation of all channels
            for ch in range(image.shape[-1]):
                image[...,ch] = (image[...,ch] - image[...,ch].mean()) / image[...,ch].std()
            # image = (image - image.mean()) / image.std()
        
        if self.avg_23:
                image = np.dstack([image, ((image[...,0] + image[...,1]) / 2)[..., np.newaxis]])
            

        return image
    
    def __get_output(self, label):
        return tf.keras.utils.to_categorical(label, num_classes=self.n_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_batch = [self.__get_input(row[1]) for row in batches.iterrows()]
        y_batch = [self.__get_output(row) for row in batches['class']]

        return np.asarray(X_batch), np.asarray(y_batch)
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)      
          
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size