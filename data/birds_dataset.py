"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as T
# from data.image_folder import make_dataset
from PIL import Image
import torch
import torch.nn.functional as F
import glob
from collections import OrderedDict
import numpy as np

class allow_unbatched(object):
    def __init__(self, input_correspondences):
        self.input_correspondences = \
            OrderedDict(input_correspondences)

    def __call__(self, f):
        def wrapped(*args, **kwargs):
            args = list(args)
            to_unbatch = []
            for inp_i, out_is in \
                    self.input_correspondences.items():
                inp = args[inp_i]
                assert len(inp.shape) in [3, 4]
                is_batched = len(inp.shape)==4
                if not is_batched:
                    args[inp_i] = inp.unsqueeze(0)
                    to_unbatch += out_is
            ret = f(*args, **kwargs)
            if type(ret) is not tuple:
                ret = (ret,)
            ret = list(ret)
            for out_i in to_unbatch:
                ret[out_i] = ret[out_i].squeeze(0)
            return tuple(ret) if len(ret) > 1 else ret[0]
        return wrapped


BIRD_IMG_DIM, BIRD_ATT_DIM, BIRD_ATT_SCALES = 128, 16, [1, 2, 4, 8]

def sample_scale(deterministic_seed=None):
    # sampler from prior over scale
    # probability proportional to 1/scale
    p = np.array([1/s for s in BIRD_ATT_SCALES])
    p = p/sum(p)
    return np.random.choice(len(BIRD_ATT_SCALES), p=p)

@allow_unbatched({0: [0]})
def upsample(x, new_size=None, scaling=None):
    if new_size is None:
        H = x.shape[2]
        assert H % scaling == 0
        new_size = H // scaling
    return F.interpolate(x,
                         (new_size, new_size),
                         mode='bilinear',
                         align_corners=False)

@allow_unbatched({0: [0]})
def downsample(x, new_size=None, scaling=None):
    if scaling is None:
        H = x.shape[2]
        assert H % new_size == 0
        scaling = H // new_size
    elif scaling == 1:
        return x
    return F.avg_pool2d(x, stride=scaling,
                        kernel_size=scaling)

def sample_bird_glimpse_location():
    # sample patch scale
    scale_i = sample_scale()
    scale = BIRD_ATT_SCALES[scale_i]
    # sample grid position
    grid_dim = BIRD_IMG_DIM - scale*BIRD_ATT_DIM
    x, y = np.random.randint(0, grid_dim+1,
                             size=2)
    return x, y, scale

def reconstruct_obs(image, sequence):
    x = torch.randn(3, 128, 128)*0.1  # random noise on unobserved parts to add stochasticity
    sequence = sorted(sequence,
                      key=lambda l: -l[2])
    for r, c, scale in sequence:
        patch = get_observed_patch(image,
                                   r, c, scale, 16)
        side_dim = 16*scale
        patch = F.interpolate(
            patch.unsqueeze(0),
            size=(side_dim, side_dim)
        ).squeeze(0)
        x[:, r:r+side_dim, c:c+side_dim] = patch
    return x

@allow_unbatched({0: [0]})
def get_observed_patch(images, R, C, scale, att_dim,
                       horizontal_flip=False):
    # handle downsampling
    B, K, H, W = images.shape
    scaled_R = R // scale
    scaled_C = C // scale
    images = downsample(images,
                        scaling=scale)

    # find coordinates to take
    rows = torch.arange(scaled_R, scaled_R+att_dim).long()
    columns = torch.arange(scaled_C, scaled_C+att_dim).long()
    if horizontal_flip:
        width = images.shape[3]
        columns = width-1-columns

    # select pixels on GPU (if using it)
    rows = rows.to(images.device)
    columns = columns.to(images.device)
    return images\
        .index_select(2, rows)\
        .index_select(3, columns)

def sample_bird_glimpse_sequence():
    T = 4
    t = np.random.randint(1, T)
    return [sample_bird_glimpse_location()
            for _ in range(t)]


class BirdsDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this\
        flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        print("Running BirdsDataset.modify_commandline_options")
        parser.add_argument('--new_dataset_option', type=float, default=1.0,
                            help='new dataset option')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass\
        of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        image_files = f"datasets/birds/{opt.phase}/images/*"
        self.image_paths = sorted(glob.glob(image_files))
        # define the default transform function. You can use <base_dataset.get_transform>;
        # You can also define your custom transform function
        #self.transform = get_transform(opt)
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

    def mask_out(self, image):
        sequence = sample_bird_glimpse_sequence()
        return reconstruct_obs(image, sequence)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself\
        and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such\
        as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.image_paths[index]
        data_B = self.transform(Image.open(path))\
                     .expand(3, -1, -1)  # convert any grayscales to RGB
        data_A = self.mask_out(data_B)
        return {'A': data_A, 'B': data_B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
