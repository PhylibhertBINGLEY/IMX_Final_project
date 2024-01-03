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

## my module and functions 
from cycleganModules import BaseOptions
from cycleganModules import TestOptions
from loadModelsCycleGAN import download_cyclegan_model



def cycleGAN(image, model_without_pretrained):

    model_name = model_without_pretrained + '_pretrained'

    download_cyclegan_model(model_without_pretrained)

    # Run the pip install command
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


    # Convert the image to a PyTorch tensor
    image_tensor = TF.to_tensor(image).unsqueeze(0)


    # Create an instance of TestOptions
    opt = TestOptions()

    # Manually set the options
    opt.dataroot = 'datasets/testStyle'  # Provide a default value or use the actual path
    #opt.test_dataroot = dataroot
    opt.name = model_name ## pretrained model name
    opt.model = 'test'
    opt.no_dropout = True
    opt.dataset_mode = 'single'  # Provide the appropriate dataset mode
    opt.max_dataset_size = 100  # Set the value according to your requirements
    opt.direction = 'BtoA'  # or 'BtoA', depending on your use case
    opt.output_nc = 3  # adjust the value based on the number of channels in your output image
    opt.input_nc = 3
    opt.preprocess = 'resize'  # adjust the value based on your desired preprocessing method
    opt.load_size = 256  # adjust the value based on your desired load size
    opt.no_flip = True  # or False, depending on your preference
    opt.batch_size = 1  # Set the batch size according to your requirements
    opt.serial_batches = True  # Set according to your requirements
    opt.num_threads = 4  # Set according to your requirements

    # Set isTrain to False for testing
    opt.isTrain = False
    opt.gpu_ids = [0]
    opt.checkpoints_dir = './checkpoints' ## pretrained model path
    #opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.name) 
    opt.model_suffix = ''  # Example: Set the model suffix if needed
    opt.ngf = 64
    opt.netG = 'resnet_9blocks'
    opt.norm = 'instance'
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    opt.load_iter = 100
    opt.verbose = 'store_true'
    opt.num_test = float("inf") # process all available images ; set the value to 1 if you wanna process only 1 image

    opt.epoch = 5


    # Parse the options
    opt.parse()
    # Create a fake dataset with a single image
    dataset = [{'A': image_tensor, 'A_paths': 'fake_path'}]

    model = create_model(opt)
    model.setup(opt)

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        generated_image = visuals['fake'][0]
        generated_image_pil = TF.to_pil_image((generated_image + 1) / 2)

        # Save the image to the folder
        save_path = f'./generated_images/image_{i}.png'
        generated_image_pil.save(save_path, format='PNG')

        # Display the generated image using matplotlib
        img_with_style = np.array(generated_image_pil)

    return img_with_style
