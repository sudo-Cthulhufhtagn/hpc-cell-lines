import pandas as pd
from helpers.config import *
import cv2
from joblib import Memory
from skimage.filters import threshold_local
from helpers.blobs import define_detector
import numpy as np
import tqdm
from typing import List, Tuple
import os 
from hydra import utils
import re
from icecream import ic
neg = lambda x: x if x<0 else 0
relu = lambda x: x if x>0 else 0

detector = define_detector()
memory = Memory(utils.to_absolute_path('cac'), verbose=0)

@memory.cache
def blobs_cutter(path, normalize_bw=False):
    im = cv2.imread(path, -1)
    if normalize_bw: # TODO: pars.normalize_bw replace with normalize_bw
        im = im / im.max() * 255
        im = im.astype('uint8')
    thr = threshold_local(im, 301, 'gaussian', offset = 0)
    im = (im > thr).astype(np.uint8)*255
    # do the blob detection
    keypoints = detector.detect(im)
    
    return [list(map(int, p.pt)) for p in keypoints]



def get_annots(path):
    """Craetes dataset of files and classes.

    Returns:
        rows of original dataset, class of drug
    """
    
    df = pd.read_csv(path, delimiter='\t', skiprows=2)
    df = df.iloc[:-2]
    df['Row'] = df['Row'].astype('int')
    df['Column'] = df['Column'].astype('int')
    
    ch1 = df[
    (df['Channel']==1) & (
    ((df['Column'].isin(positions[0][1])) &
    (df['Row'].isin(positions[0][0]))) |
    ((df['Column'].isin(positions[1][1])) &
    (df['Row'].isin(positions[1][0]))) |
    ((df['Column'].isin(positions[2][1])) &
    (df['Row'].isin(positions[2][0]))) |
    ((df['Column'].isin(positions[3][1])) &
    (df['Row'].isin(positions[3][0]))))
    ].copy()
    
    # tr0 = ((ch1['Column'].isin(positions[0][1])) &
    #         (ch1['Row'].isin(positions[0][0])))*0
    # tr0[((ch1['Column'].isin(positions[1][1])) &
    #         (ch1['Row'].isin(positions[1][0])))] = 1
    # tr0[((ch1['Column'].isin(positions[2][1])) &
    #         (ch1['Row'].isin(positions[2][0])))] = 2
    # tr0[((ch1['Column'].isin(positions[3][1])) &
    #         (ch1['Row'].isin(positions[3][0])))] = 3
    
    ch1.loc[:,'class'] = 0
    # ch1.loc[((df['Column'].isin(positions[1][1])) &
    #         (df['Row'].isin(positions[1][0]))),]['class'] = 1
    
    ch1.loc[(ch1['Column'].isin(positions[1][1])) & (ch1['Row'].isin(positions[1][0])), 'class'] = 1
    
    ch1.loc[(ch1['Column'].isin(positions[2][1])) & (ch1['Row'].isin(positions[2][0])), 'class'] = 2
    
    ch1.loc[(ch1['Column'].isin(positions[3][1])) & (ch1['Row'].isin(positions[3][0])), 'class'] = 3
    
    return ch1#, tr0

def impreprocessor(paths: list, pars: Parametrizer, crop_list: list, save_path, label) -> np.ndarray:
    """
    Read images from paths and return numpy array of shape (len(paths), *input_shape, n_channels)
    """
    # if not pars.channel_3_avg_12:
    #     assert len(paths) == pars.n_channels, 'Number of images to load must be equal to number of channels'
        
    # craete np array with zeros of shape (len(paths), *input_shape, n_channels)
    images_crop = np.zeros((len(crop_list), *pars.input_shape, pars.n_channels))
    images_crop = np.zeros((len(crop_list), *pars.input_shape, 7))
    image = np.zeros((*pars.image_shape, len(paths)), )#dtype='uint16')
    # print(image.dtype)
    
    for i, path in enumerate(paths):
        img = cv2.imread(path, -1)
        if pars.normalize_bw_non_one:
            img = (img-img.min()) / img.max() * 255
            img = img.astype('uint8')
        
        image[...,i] = img
        
    # if pars.channel_3_avg_12:
    #     image[...,2] = (image[...,0] + image[...,1]) / 2
        
    # if pars.normalize_color:
    #     # using mean and standard deviation of all channels
    #     image = (image - image.mean()) / image.std()
    #     if pars.convert_to_uint8:
    #         image  = (image - image.min()) / image.max() * 255
    #         image = image.astype('uint8')

    max_x = pars.image_shape[0]
    max_y = pars.image_shape[1]
    
    for i, keypoints in enumerate(crop_list):
        # x, y = list(map(int, keypoints.pt))
        x, y = keypoints[0], keypoints[1]
        d = pars.d
        start = (
            relu(x - d) - relu(x+d-max_x), 
            min(y + d, max_y) + relu(d-y)
        )
        stop = (
            min(x+d, max_x) + relu(d-x),
            relu(y-d) - relu(y+d-max_y)
        )
        if pars.padding == 'value':
            value = pars.padding_value#(image)
            # put image to the center of the image of shape pars.input_shape and surround it by value
            # chech if the image is not bigger than pars.input_shape both x and y axis
            if (stop[1]-start[1]) > pars.input_shape[0] or (stop[0]-start[0]) > pars.input_shape[1]:
                # do rescailing down to pars.input_shape
                images_crop[i, ...] = cv2.resize(image[stop[1]:start[1], start[0]:stop[0], :], pars.input_shape)
                continue
                
            images_crop[i, ...] = value
            row_width = (start[1]-stop[1]) // 2
            col_width = (stop[0]-start[0]) // 2
            images_crop[i, 
                        pars.input_shape[0]//2-row_width:pars.input_shape[0]//2+row_width,
                        pars.input_shape[1]//2-col_width:pars.input_shape[1]//2+col_width,
                        ] = image[stop[1]:start[1], start[0]:stop[0], :]
        elif pars.padding == 'rescale':
            # rescale image to fit pars.input_shape
            images_crop[i, ...] = cv2.resize(image[stop[1]:start[1], start[0]:stop[0], :], pars.input_shape)
            
        # images_crop[i, ...] = image[stop[1]:start[1], start[0]:stop[0], :]
    
    list_of_paths = []
    for id, img in enumerate(images_crop):
        # get file name without extension from path
        for ch in range(len(paths)):
            path_c = re.sub(r'-ch0\d', f'-ch0{ch+1}', path)
            save_path_n = os.path.join(save_path, os.path.basename(path_c).split('.')[0]) + '_id{}_cl{}.png'.format(id, label)
            list_of_paths.append(save_path_n)
            cv2.imwrite(save_path_n, img[...,ch])
        
    return list_of_paths

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

def create_dataset(X, y, pars, top, cwd) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset for training.
    #TODO: replace top
    """
    images_train = []
    labels_train = []
    channels = pars.channels
    dset_path = pars.dataset_path
    dset_name = 'i{}_d{}_pad{}'.format(pars.input_shape[0], pars.d, pars.padding)
    dset_path = os.path.join(cwd, dset_path, dset_name)
    if mkdir(dset_path):
        for i in tqdm.tqdm(range(len(X))):
            row = X.iloc[i]
            label = y.iloc[i]
            if row['Channel'] == pars.segmentation_channel:
                seg_path = os.path.join(top, row['__URL'])
                # print(seg_path, top + row['__URL'])
                keypoints = blobs_cutter(seg_path, pars.normalize_bw)
                # find 'ch0' in the string and replace symbol after 'ch'
                channel_path = seg_path.replace('ch01', 'ch0{}')
                channels_paths = [channel_path.format(i) for i in range(1,8)]
                im_paths = impreprocessor(channels_paths, pars, keypoints, dset_path, label)
                images_train.extend(im_paths)
                labels_train.extend([label]*len(keypoints))
                # for indx in positions:
                #     if row['Row'] in positions[indx][0] and row['Column'] in positions[indx][1]:
                #         labels.extend([indx]*len(keypoints))
                #         break
                break # TODO: remove this shit
    else:
        for file in os.listdir(dset_path):
            # check if filename is matching at least part of the string from train_X['__URl]
            
            if X['__URL'].str.contains(file.split('-ch')[0], case=False).any():
                images_train.append(os.path.join(dset_path, file))
                match = re.search(r'_cl\d', file).group()
                # ic(match)
                labels_train.append(int(match[-1]))
        
        # just load folders which do already exist
    # for i in tqdm.tqdm(range(len(train_X))):
    #     row = train_X.iloc[i]
    #     label = train_y.iloc[i]
    #     if row['Channel'] == pars.segmentation_channel:
    #         seg_path = os.path.join(top, row['__URL'])
    #         print(seg_path, top + row['__URL'])
    #         keypoints = blobs_cutter(seg_path, pars.normalize_bw)
    #         # find 'ch0' in the string and replace symbol after 'ch'
    #         channel_path = seg_path.replace('ch01', 'ch0{}')
    #         channels_paths = [channel_path.format(i) for i in channels]
    #         images_train.extend(imreader(channels_paths, pars, keypoints))
    #         labels_train.extend([label]*len(keypoints))
    #         # for indx in positions:
    #         #     if row['Row'] in positions[indx][0] and row['Column'] in positions[indx][1]:
    #         #         labels.extend([indx]*len(keypoints))
    #         #         break
    #         break
        
    n_cls = 4
    # ic(X['__URL'].str.contains(os.listdir(dset_path)[0].split('_')[0]).any())
    # ic(os.listdir(dset_path)[0].split('_')[0])
    # ic(labels_train)
    labels_train = np.array(labels_train)
    one_hot = np.zeros((labels_train.size, n_cls), dtype=int)
    # ic(labels_train)
    one_hot[np.arange(labels_train.size), labels_train] = 1
    
        
    return images_train, one_hot


# def imreader(paths: list, pars: Parametrizer, crop_list: list) -> np.ndarray:
#     """
#     Read images from paths and return numpy array of shape (len(paths), *input_shape, n_channels)
#     """
#     if not pars.channel_3_avg_12:
#         assert len(paths) == pars.n_channels, 'Number of images to load must be equal to number of channels'
        
#     # craete np array with zeros of shape (len(paths), *input_shape, n_channels)
#     images_crop = np.zeros((len(crop_list), *pars.input_shape, pars.n_channels))
#     image = np.zeros((*pars.image_shape, len(paths)), )#dtype='uint16')
#     # print(image.dtype)
    
#     for i, path in enumerate(paths):
#         img = cv2.imread(path, -1)
#         if pars.normalize_bw_non_one:
#             img = (img-img.min()) / img.max() * 255
#             img = img.astype('uint8')
        
#         image[...,i] = img
        
#     if pars.channel_3_avg_12:
#         image[...,2] = (image[...,0] + image[...,1]) / 2
        
#     if pars.normalize_color:
#         # using mean and standard deviation of all channels
#         image = (image - image.mean()) / image.std()
#         if pars.convert_to_uint8:
#             image  = (image - image.min()) / image.max() * 255
#             image = image.astype('uint8')

#     max_x = pars.image_shape[0]
#     max_y = pars.image_shape[1]
    
#     for i, keypoints in enumerate(crop_list):
#         # x, y = list(map(int, keypoints.pt))
#         x, y = keypoints[0], keypoints[1]
#         d = pars.d
#         start = (
#             relu(x - d) - relu(x+d-max_x), 
#             min(y + d, max_y) + relu(d-y)
#         )
#         stop = (
#             min(x+d, max_x) + relu(d-x),
#             relu(y-d) - relu(y+d-max_y)
#         )
#         if pars.padding == 'value':
#             value = pars.padding_value#(image)
#             # put image to the center of the image of shape pars.input_shape and surround it by value
#             # chech if the image is not bigger than pars.input_shape both x and y axis
#             if (stop[1]-start[1]) > pars.input_shape[0] or (stop[0]-start[0]) > pars.input_shape[1]:
#                 # do rescailing down to pars.input_shape
#                 images_crop[i, ...] = cv2.resize(image[stop[1]:start[1], start[0]:stop[0], :], pars.input_shape)
#                 continue
                
#             images_crop[i, ...] = value
#             row_width = (start[1]-stop[1]) // 2
#             col_width = (stop[0]-start[0]) // 2
#             images_crop[i, 
#                         pars.input_shape[0]//2-row_width:pars.input_shape[0]//2+row_width,
#                         pars.input_shape[1]//2-col_width:pars.input_shape[1]//2+col_width,
#                         ] = image[stop[1]:start[1], start[0]:stop[0], :]
#         elif pars.padding == 'rescale':
#             # rescale image to fit pars.input_shape
#             images_crop[i, ...] = cv2.resize(image[stop[1]:start[1], start[0]:stop[0], :], pars.input_shape)
            
#         # images_crop[i, ...] = image[stop[1]:start[1], start[0]:stop[0], :]
        
#     return images_crop