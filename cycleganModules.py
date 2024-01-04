from data import create_dataset
from models import create_model
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
# Import necessary modules
import argparse
import os
import glob
import subprocess
import requests





class BaseOptions:
    def initialize(self, parser):
        parser.add_argument('--dataroot', required=False, default=None, help='path to dataset')
        # ... other existing code
        return parser


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #parser.add_argument('--test_name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--test_dataroot', required=False, help='path to the test dataset')
        parser.add_argument('--dataset_mode', type=str, default='single', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'), help='maximum number of samples allowed per dataset. If the dataset directory contains more than this number, only a subset is loaded.')
        parser.add_argument('--direction', type=str, default='AtoB', help='specify the direction of the transformation: AtoB or BtoA')
        parser.add_argument('--output_nc', type=int, default=3, help='number of channels in the output image')
        parser.add_argument('--input_nc', type=int, default=3, help='number of input channels for the generator')
        parser.add_argument('--preprocess', type=str, default='resize', help='specify the preprocessing method: none | resize | ... (add more as needed)')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--serial_batches', action='store_true', help='if specified, do not shuffle the data')
        parser.add_argument('--num_threads', type=int, default=4, help='number of threads for loading data')
        # Manually add isTrain attribute and set it to False
        parser.add_argument('--isTrain', action='store_false', help='set isTrain to False for testing')
        parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids. e.g., --gpu_ids 0 1 2')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--model_suffix', type=str, default='', help='suffix to add to the model for saving and loading')  # Add this line
        parser.add_argument('--ngf', type=int, default=64, help='number of generator filters in the last conv layer')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify the generator architecture')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization type [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--load_iter', type=int, default=0, help='which iteration to load for testing')
        parser.add_argument('--epoch', type=int, default=0, help='which epoch to load for testing')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')


        return parser

    def parse(self):
        # You can leave this method empty for now or add custom logic if needed
        pass