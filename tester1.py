import cv2
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import mlflow
import hydra
from hydra import utils
import optuna
from helpers.helpers import get_annots, create_dataset
from helpers.config import *
from sklearn.model_selection import train_test_split
from icecream import ic
#%%
import re
#cut everything after _ch including _ch
# re.sub(r'_ch\d.*', '', 'image_ch1.png')
# g = 'dfnxjkb,-ch09.gg'
# p= 'xjkbfhbg-ch08gg.t'
# for i in range(3):
#     g = re.sub(r'-ch0\d', f'-ch0{i}', p)
    # print(p,g)
# re.search(r'_ch\d', 'image_ch1.png').group()
# re.

#%%
# print debug for gpu
print(f"[CHECK]: GPU - {tf.config.list_physical_devices('GPU')}")

def image_generator(filenames, labels, channels, normalize_color=False):
    for filename, label in zip(filenames, labels):
        # ic(filename, label)
        filename = filename.decode('utf-8')
        # check if filename contains channel in format '_chX' with regex and matches channel 0
        if re.search(r'-ch0\d', filename).group() == f'-ch0{channels[0]}':
            imgs = []
            for i in channels:
                # replace channel in filename with current channel
                filename = re.sub(r'-ch0\d', f'-ch0{i}', filename)
                # read cv2 image
                imgs.append(cv2.imread(filename, -1))
                # print(filename)
            
            # stack images along as colors
            # for i in range(len(imgs)):
            #     ic(imgs[i].shape)
            image = np.stack(imgs, axis=-1)
            
            if normalize_color:
                # using mean and standard deviation of all channels
                for ch in range(image.shape[-1]):
                    image[...,ch] = (image[...,ch] - image[...,ch].mean()) / image[...,ch].std()
                # image = (image - image.mean()) / image.std()
                
            # ic(type(image), type(label), type(label[0]), type(image[0]))
            
            yield image, label

@hydra.main(config_path="conf", config_name="config")
def main(conf):
    global top
    cfg = conf.basic
    top = os.path.join(hydra.utils.get_original_cwd(),
                       top)
    # np.random(42)
    # ic(cfg.channels, cfg.channels_lists[cfg.channels])
    dataset, classes = get_annots(os.path.join(top, 'image.index.txt'))
    
    cwd = utils.get_original_cwd()
    create_dataset(dataset, classes, cfg, top, cwd)
    train_X, test_X, train_y, test_y = train_test_split(dataset, 
                                                        classes, 
                                                        test_size=cfg.splits.test, 
                                                        random_state=42, 
                                                        stratify=classes,
                                                        shuffle=True, )
    # split also to train and val
    
    # images_train, labels_train = create_dataset(train_X, train_y, cfg, top)
    # train_X, val_X, train_y, val_y = train_test_split(train_X, 
    #                                                   train_y, 
    #                                                   test_size=0.2, 
    #                                                   random_state=42, 
    #                                                   # stratify=classes, # TODO: uncomment
    #                                                   shuffle=False, )
    
    if cfg.model == 'ResNet50':
        from helpers.model_factory.res_net import get_model
    
    train_images, val_images, train_labels, val_labels = train_test_split(train_X, train_y, test_size=cfg.splits.val, random_state=42, stratify=train_y)
    images_train, labels_train = create_dataset(train_images, train_labels, cfg, top, cwd)
    # images_train, labels_train = create_dataset(dataset, classes, cfg, top, cwd)
    val_images, val_labels = create_dataset(val_images, val_labels, cfg, top, cwd)
    train_dataset = tf.data.Dataset.from_generator(
        image_generator,
        output_signature=(
            tf.TensorSpec(shape=(*cfg.input_shape, cfg.n_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(4), dtype=tf.uint8)  # Assuming labels are strings
        ),
        args=(images_train, labels_train, cfg.channels_lists[cfg.channels], cfg.normalize_ch_color)
    )
    
    # val_dataset = train_dataset
    val_dataset = tf.data.Dataset.from_generator(
        image_generator,
        output_signature=(
            tf.TensorSpec(shape=(*cfg.input_shape, cfg.n_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(4), dtype=tf.uint8)  # Assuming labels are strings
        ),
        args=(val_images, val_labels, cfg.channels_lists[cfg.channels], cfg.normalize_ch_color)
    )
    print(f"Length train: {len(images_train)} val: {len(val_images)} classes tr: {labels_train.sum(axis=0)} val: {val_labels.sum(axis=0)}")
    
    # val_dataset = train_dataset.take(int(len(train_dataset)*(1-train_percentage)))
    # train_dataset = train_dataset.skip(int(len(train_dataset)*train_percentage))
    # train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    # print(f"Dataset sizes: train - {len(train_dataset)}, val - {len(val_dataset)}")
    model = get_model(cfg)
    # tf.config.experimental_run_functions_eagerly(True)
    batch_size = cfg.batch_size
    
    # iterate over tf dataset 
    # for i in train_dataset.batch(batch_size).prefetch(1):
    #     ic(i)
    
    # save images train 
    # with open('images_train.txt', 'w') as f:
    #     for item in images_train:
    #         f.write("%s\n" % item)
    
    # iterator = iter(image_generator(images_train, labels_train, cfg.channels, True), )
    # # iterate over image_generator
    # for i in range(10):
    #     im, label = next(iterator)
    #     ic(im.shape, im.dtype, label)
        # cv2.imwrite('test.png', im)
    # cv2.imwrite('test.png', im)
    # ic(type(next(iterator)[1][0]))
    
    mlflow.set_tracking_uri('file://' + cwd + '/mlruns')
    # create new experiment or continue if exists
    mlflow.set_experiment(cfg.experiment_name)
    
    with mlflow.start_run():
        # run mlflow
        checkpoint_filepath = 'checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        
        # mlflow.tensorflow.autolog()
        history = model.fit(
            train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), 
            batch_size=batch_size, 
            epochs=cfg.epochs,
            # callbacks=[model_checkpoint_callback],
            validation_data=val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
            )
        
        keys = history.history.keys()
        monitor = "val_accuracy"
        
        mlflow.log_param('model', cfg.model)
        mlflow.log_params(cfg)
        mlflow.log_artifacts(utils.to_absolute_path('conf'))
        mlflow.tensorflow.log_model(model, 'model')
        # get os git hash commit
        mlflow.log_param('git_commit', os.popen('git rev-parse HEAD').read().strip())
        # mlflow.log_metric('val_accuracy', history.history['val_accuracy'][-1])
        
        # Example usage:
        for key in keys:
            print(f'{key} - {history.history[key]}')
            for i in range(len(history.history[key])):
                mlflow.log_metric(key, history.history[key][i], step=i)
        
        # return history.history[monitor][-1]
    
if __name__=='__main__':
    # cfg = Parametrizer()
    main()


