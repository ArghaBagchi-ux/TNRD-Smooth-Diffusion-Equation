import torch
import glob
import os
import random
from torchvision import transforms
from PIL import Image
import numpy as np

class DatasetNoise(torch.utils.data.dataset.Dataset):
    def __init__(self, root='/content/denoising/data/FoETrainingSets180/', noise_sigma=50., training=True, crop_size=60, blind_denoising=False, gray_scale=False, max_size=None):
        self.root = root
        self.noise_sigma = noise_sigma
        self.training = training
        self.crop_size = crop_size
        self.blind_denoising = blind_denoising
        self.gray_scale = gray_scale
        self.max_size = max_size

        self._init()

    def _init(self):
        # data paths
        targets = glob.glob(os.path.join(self.root, '*.*'))[:self.max_size]
        self.paths = {'target' : targets}

        # transforms
        t_list = [transforms.ToTensor()]
        self.image_transform = transforms.Compose(t_list)

    def _get_augment_params(self, size):
        random.seed(random.randint(0, 12345))

        # position
        w_size, h_size = size
        x = random.randint(0, max(0, w_size - self.crop_size))
        y = random.randint(0, max(0, h_size - self.crop_size))

        # flip
        flip = random.random() > 0.5
        return {'crop_pos': (x, y), 'flip': flip}

    def _augment(self, image, aug_params):
        x, y = aug_params['crop_pos']
        image = image.crop((x, y, x + self.crop_size, y + self.crop_size))
        if aug_params['flip']:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, index):
        # target image
        if self.gray_scale:
            target = Image.open(self.paths['target'][index]).convert('L')
        else:
            target = Image.open(self.paths['target'][index]).convert('RGB')

        # transform
        if self.training:
            aug_params = self._get_augment_params(target.size)
            target = self._augment(target, aug_params)
        target = self.image_transform(target) * 255

        # add multiplicative gamma noise
        #if self.blind_denoising:
            #noise_sigma = random.randint(0, self.noise_sigma)
        #else:
            #noise_sigma = self.noise_sigma
        L = 2
        a = L
        b = 1 / L
        #shape = target.shape
        gamma_shape = target.shape # the shape of the gamma noise should match the image
        gamma_noise = torch.from_numpy(np.random.gamma(shape=a, scale=b, size=gamma_shape)).float()
        #shape = target.shape
        #gamma_shape = shape  # the shape of the gamma noise should match the image
        #gamma_noise = torch.from_numpy(np.random.gamma(shape=1.0, scale=noise_sigma / 255.0, size=gamma_shape)).float()

        input = target * gamma_noise

        return {'input': input, 'target': target, 'path': self.paths['target'][index]}

    def __len__(self):
        return len(self.paths['target'])

# Usage
# dataset = DatasetNoise(root='path_to_your_dataset', noise_sigma=50., training=True, crop_size=60, blind_denoising=False, gray_scale=False, max_size=None)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
