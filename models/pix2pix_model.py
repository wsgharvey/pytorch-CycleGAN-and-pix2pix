import torch
from .base_model import BaseModel
from . import networks

import numpy as np
from collections import OrderedDict
from scipy.ndimage.morphology import binary_erosion
import torch.nn.functional as F

def build_mask(shape, att_shape, locations=[]):
    mask = torch.zeros(shape)
    for r, c in locations:
        mask[r:r+att_shape[0],
             c:c+att_shape[1]] = 1
    return mask

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

@allow_unbatched({0: [0]})
def erode(binary_image, erosion=1):
    """
    Sets 1s at boundaries of binary_image to 0
    """
    batch_array = binary_image.data.cpu().numpy()
    return torch.tensor(
        np.stack([
            binary_erosion(
                array,
                iterations=erosion,
                border_value=1,  # so that we don't get border of zeros
            ).astype(array.dtype)
            for array in batch_array])
    ).to(binary_image.device)

BIRD_ATT_SCALES = [1, 2, 4, 8]
BIRD_ATT_DIM = 16
BIRD_IMG_DIM = 128

@allow_unbatched({0: [0], 1: []})
def embed(glimpsed_images,
          completion_images,
          locations,
          erosion=1):
    """
    images can be batched, locations cannot be
    """
    device = glimpsed_images.device
    locations = [(scale, [(x, y) for x, y, s in
                          locations if s == scale])
                 for scale in BIRD_ATT_SCALES]
    inputs = []
    for scale, coords in locations:
        completion = downsample(completion_images,
                                scaling=scale)
        if len(coords) == 0:
            morphed = torch.cat(
                [completion,
                 completion[:, :1]*0.],  # mask channel
                dim=1)
        else:
            B, _, H, W = completion.shape
            glimpsed = downsample(glimpsed_images,
                                  scaling=scale)
            scaled_coords = [(r//scale, c//scale)
                             for r, c in coords]
            obs_mask = build_mask(
                shape=(H, W),
                att_shape=(BIRD_ATT_DIM,
                           BIRD_ATT_DIM),
                locations=scaled_coords
            ).unsqueeze(0)\
            .unsqueeze(0)\
            .to(device)  # batch + channel dims
            # erode each masks to make some zeros:
            eroded_obs_mask = erode(obs_mask,
                                    erosion).to(device)
            eroded_completion_mask = erode(1-obs_mask,
                                           erosion).to(device)
            morphed = (completion *\
                       eroded_completion_mask) + \
                       (glimpsed *\
                        eroded_obs_mask)
            morphed = torch.cat([
                morphed,
                obs_mask.expand(B, -1, -1, -1)],
                                dim=1)
        inputs.append(
            upsample(morphed, new_size=BIRD_IMG_DIM)
        )
    return torch.cat(inputs, dim=1)

def sample_scale(deterministic_seed=None):
    # sampler from prior over scale
    # probability proportional to 1/scale
    p = np.array([1/s for s in BIRD_ATT_SCALES])
    p = p/sum(p)
    return np.random.choice(len(BIRD_ATT_SCALES), p=p)


def sample_bird_glimpse_location():
    # sample patch scale
    scale_i = sample_scale()
    scale = BIRD_ATT_SCALES[scale_i]
    # sample grid position
    grid_dim = BIRD_IMG_DIM - scale*BIRD_ATT_DIM
    x, y = np.random.randint(0, grid_dim+1,
                             size=2)
    return x, y, scale

def sample_bird_glimpse_sequence():
    T = 4
    t = np.random.randint(1, T)
    return [(32, 32, 4)] + \
           [sample_bird_glimpse_location()
            for _ in range(t-1)]



class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--disc-padding', type=int, default=0, help='padding around edges of glimpses to make it harder for discriminator')
            parser.add_argument('--ours', type=bool, default=True, help='Whether to use our embedding for discriminator.')
            parser.add_argument('--embedding_nc', type=int, default=16, help='Num channels in fancy glimpse embedding.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.erosion = opt.disc_padding
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.embedding_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.embedding_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.sequences = [sample_bird_glimpse_sequence() for _ in range(self.real_A.shape[0])]
        #input['sequences']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        embedding = []
        for real, sequence in zip(self.real_B,
                                  self.sequences):
            noise_image = torch.randn(real.shape).to(real.device)
            embedding.append(
                embed(real, noise_image,
                      sequence, erosion=self.erosion)
            )
        embedding = torch.stack(embedding, dim=0)
        self.fake_B = self.netG(embedding)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        embedding_real = []
        embedding_fake = []
        for real, fake, sequence in zip(self.real_B,
                                        self.fake_B,
                                        self.sequences):
            embedding_real.append(
                embed(real, real, sequence, erosion=self.erosion)
            )
            embedding_fake.append(
                embed(real, fake, sequence, erosion=self.erosion)
            )
        embedding_real = torch.stack(embedding_real, dim=0)
        embedding_fake = torch.stack(embedding_fake, dim=0)

        pred_fake = self.netD(embedding_fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD(embedding_real)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        embedding_fake = []
        for real, fake, sequence in zip(self.real_B,
                                        self.fake_B,
                                        self.sequences):
            embedding_fake.append(
                embed(real, fake, sequence, erosion=self.erosion)
            )
        embedding_fake = torch.stack(embedding_fake, dim=0)

        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(embedding_fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
