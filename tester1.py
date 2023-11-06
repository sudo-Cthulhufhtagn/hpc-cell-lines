import cv2
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import mlflow
import hydra
from hydra import utils
import optuna
from helpers.helpers import get_annots, create_dataset, inflate
from helpers.config import *
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from helpers.tf_helper import CustomDataGen
from icecream import ic
#%%
import re
print(f"[CHECK]: GPU - {tf.config.list_physical_devices('GPU')}")
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

def image_generator(filenames, labels, channels, normalize_color=False, channel_3_avg_12=False):
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
            
            if channel_3_avg_12:
                image = np.dstack([image, ((image[...,0] + image[...,1]) / 2)[..., np.newaxis]])
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
    print("Loading annotations")
    dataset = get_annots(os.path.join(top, 'image.index.txt'))
    print("Loading train dataset")
    # train_set = inflate(dataset.iloc[0:1], top)
    # test_set = dataset.iloc[0:1]
    train_set, test_set = train_test_split(dataset, test_size=cfg.splits.test, random_state=42, stratify=dataset['class'])
    train_set, val_set = train_test_split(train_set, test_size=cfg.splits.val, random_state=42, stratify=train_set['class'])
    train_set = inflate(train_set, top)
    val_set = inflate(val_set, top)
    print("Inflation finished")
    class_counts = train_set['class'].value_counts()
    min_class_count = class_counts.min()
    downsampled_data = pd.DataFrame()

    # Iterate over unique classes and downsample each class to the minimum class count
    for class_label in class_counts.index:
        class_data = train_set[train_set['class'] == class_label]
        downsampled_class_data = resample(class_data, replace=False, n_samples=min_class_count, random_state=42)
        downsampled_data = pd.concat([downsampled_data, downsampled_class_data])
    print(f"Length train: {len(train_set)}/{cfg.batch_size}:{len(downsampled_data)} classes tr: {train_set['class'].value_counts()} ")#val: {val_labels.sum(axis=0)}")
        
    train_set = downsampled_data
    
    train_loader = CustomDataGen(train_set,
                                 batch_size=cfg.batch_size,
                                 top_path=top,
                                 avg_23=cfg.channel_3_avg_12,
                                 normalize_ch_color=cfg.normalize_ch_color,
                                 input_size=cfg.input_shape,
                                channels=cfg.channels_lists[cfg.channels],
                                )
    print(f"Length train: {len(train_set)}/{cfg.batch_size}:{train_loader.__len__()} classes tr: {train_set['class'].value_counts()} ")#val: {val_labels.sum(axis=0)}")
    
    validation_loader = CustomDataGen(val_set,
                                 batch_size=cfg.batch_size,
                                 top_path=top,
                                 avg_23=cfg.channel_3_avg_12,
                                 normalize_ch_color=cfg.normalize_ch_color,
                                 input_size=cfg.input_shape,
                                channels=cfg.channels_lists[cfg.channels],
                                )
    
    # print(train_loader[0])
    
    cwd = utils.get_original_cwd()
    # create_dataset(dataset, classes, cfg, top, cwd, True)
    # print(f"Dataset distr: {dataset.shape}, {classes.value_counts()}")
    # train_X, test_X, train_y, test_y = train_test_split(dataset, 
    #                                                     classes, 
    #                                                     test_size=cfg.splits.test, 
    #                                                     random_state=42, 
    #                                                     stratify=classes,
    #                                                     shuffle=True, )
    # split also to train and val
    
    # images_train, labels_train = create_dataset(train_X, train_y, cfg, top)
    # train_X, val_X, train_y, val_y = train_test_split(train_X, 
    #                                                   train_y, 
    #                                                   test_size=0.2, 
    #                                                   random_state=42, 
    #                                                   # stratify=classes, # TODO: uncomment
    #                                                   shuffle=False, )
    
    if cfg.model.name == 'ResNet50':
        from helpers.model_factory.res_net import get_model
    
    # train_images, val_images, train_labels, val_labels = train_test_split(train_X, train_y, test_size=cfg.splits.val, random_state=42, stratify=train_y)
    # images_train, labels_train = create_dataset(train_images, train_labels, cfg, top, cwd)
    # images_train, labels_train = create_dataset(dataset, classes, cfg, top, cwd)
    # val_images, val_labels = create_dataset(val_images, val_labels, cfg, top, cwd)
    # train_dataset = tf.data.Dataset.from_generator(
    #     image_generator,
    #     output_signature=(
    #         tf.TensorSpec(shape=(*cfg.input_shape, cfg.n_channels), dtype=tf.float32),
    #         tf.TensorSpec(shape=(4), dtype=tf.uint8)  # Assuming labels are strings
    #     ),
    #     args=(images_train, labels_train, cfg.channels_lists[cfg.channels], cfg.normalize_ch_color, cfg.channel_3_avg_12)
    # )
    
    # val_dataset = train_dataset
    # val_dataset = tf.data.Dataset.from_generator(
    #     image_generator,
    #     output_signature=(
    #         tf.TensorSpec(shape=(*cfg.input_shape, cfg.n_channels), dtype=tf.float32),
    #         tf.TensorSpec(shape=(4), dtype=tf.uint8)  # Assuming labels are strings
    #     ),
    #     args=(val_images, val_labels, cfg.channels_lists[cfg.channels], cfg.normalize_ch_color, cfg.channel_3_avg_12)
    # )
    # print(f"Length train: {len(train_set)}:{train_loader.__len__()} val: {len(train_set)} classes tr: {train_set['class'].value_counts()} ")#val: {val_labels.sum(axis=0)}")
    # print(f"Length train: {len(train_set)}/{cfg.batch_size}:{train_loader.__len__()} classes tr: {train_set['class'].value_counts()} ")#val: {val_labels.sum(axis=0)}")
    
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
        # checkpoint_filepath = 'checkpoint'
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=True,
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_best_only=True)
        mlflow.log_params(cfg)
        mlflow.log_param('job_id', os.getenv('SLURM_JOB_ID'))
        mlflow.log_param('git_commit', os.popen('git rev-parse HEAD').read().strip())
        mlflow.log_artifacts(utils.to_absolute_path('conf'))
        
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            # min_delta=0,
            patience=2,
            verbose=0,
            mode='max',
            # baseline=None,
            restore_best_weights=True,
            start_from_epoch=3
        )
        # mlflow.tensorflow.autolog()
        history = model.fit(
            # train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), 
            train_loader,
            batch_size=batch_size, 
            epochs=cfg.epochs,
            callbacks=[early_stopping],
            validation_data=validation_loader,
            # validation_data=val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
            )
        
        keys = history.history.keys()
        monitor = "val_accuracy"
        
        # mlflow.log_param('model', cfg.model)
        mlflow.tensorflow.log_model(model, 'model')
        # get os git hash commit
        # mlflow.log_metric('val_accuracy', history.history['val_accuracy'][-1])
        
        # Example usage:
        for key in keys:
            print(f'{key} - {history.history[key]}')
            for i in range(len(history.history[key])):
                mlflow.log_metric(key, history.history[key][i], step=i)
                
        # test imagewise accuracy
        img_max_vote_soft = []
        img_max_vote_hard = []
        for row in test_set.iterrows():
            test_paths = inflate(row[1], top, row=True)
            test_loader = CustomDataGen(test_paths,
                        batch_size=1,
                        top_path=top,
                        avg_23=cfg.channel_3_avg_12,
                        normalize_ch_color=cfg.normalize_ch_color,
                        input_size=cfg.input_shape,
                    channels=cfg.channels_lists[cfg.channels],
                    )
            # model predict
            results = model.predict(test_loader, batch_size=1)
            # stack predictions
            # results_s = np.stack(results, axis=0)
            # print(results_s.shape, results.shape)
            img_max_vote_soft.append(results.sum(axis=0).argmax())
            img_max_vote_hard.append(np.bincount(results.argmax(axis=1)).argmax())
            # print(f"counts - {results.sum(axis=0)}, argmax = {results.sum(axis=0).argmax()}")
            # first find max in each row and then show the index which appears maimum number
            # print(f"{np.unique(results.argmax(axis=1), return_counts=True)}")
            
        mlflow.log_metric('test_imagewise_soft_accuracy', (np.array(img_max_vote_soft) == test_set['class'].values).sum() / len(test_set))
        mlflow.log_metric('test_imagewise_hard_accuracy', (np.array(img_max_vote_hard) == test_set['class'].values).sum() / len(test_set))
        print('predicted:', np.bincount(img_max_vote_soft), np.array(img_max_vote_hard))
            
            
            
        
        # return history.history[monitor][-1]
    
if __name__=='__main__':
    # cfg = Parametrizer()
    main()


