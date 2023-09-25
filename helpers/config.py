from dataclasses import dataclass

top = '/gpfs/space/projects/PerkinElmer/2022-09-07_CellPaintingAndBrightfield/hs/0bfb6012-5223-4b0d-ab6c-ef728112c175/images/'
# top = '/home/am/projects/tu/biomed/'
cachedir = 'cac' # cache for blobs_cutter and joblib

# name: (rows, columns)
# 0 - Palbociclib, 1 - MLN8237, 2 - AZD1152, 3 - CYC116
positions = {
    0: (list(range(6,14)), list(range(1,7))),
    1: (list(range(6,13)), list(range(7,13))),
    2: (list(range(6,13)), list(range(13,19))),
    3: (list(range(6,13)), list(range(19,25)))
}   

@dataclass
class Parametrizer():
    normalize_bw = True
    normalize_bw_non_one = False
    normalize_color = False
    convert_to_uint8 = True
    segmentation_channel = 1
    input_shape = (224, 224,)
    image_shape = (1080, 1080,)
    n_channels = 3
    padding = 'value' # or rescale
    # padding = 'rescale' # or rescale
    padding_value = lambda *_: 0 # or np.mean
    d = 80
    channel_3_avg_12 = False
    shuffle_dataset = True
    model = 'ResNet50'