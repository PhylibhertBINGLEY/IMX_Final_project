# import of our main modules
from deepMatting import deepmatting
from cycleGAN import cycleGAN

# other imports
import argparse         # for arg
from PIL import Image   # for images
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

"""
Here the helper functions
"""

# Juste pour debugger en attendant
def printArgs(image_path, pretrained_model, apply_to_foreground):
    # Votre logique de traitement ici
    print(f"Image: {image_path}")
    print(f"Pretrained Model: {pretrained_model}")
    print(f"Apply to Foreground: {apply_to_foreground}")

"""
Main:
  1) Check arguments
  2) Apply function defined in the previous part (that use our modules)
"""
if __name__ == "__main__":
    # 1] Check the args
    parser = argparse.ArgumentParser(description="Apply style to an image.")
    parser.add_argument("image", help="Path to the image you want to apply the style.")
    parser.add_argument("--pretrained_model", default="./model_0", help="Path to the pretrained style model.")
    parser.add_argument("--applyToForeground", type=bool, default=False, const=True, nargs="?", help="Apply style to foreground (default: False).")

    args = parser.parse_args()

    if not args.image:
        print("Error: Please provide the path to the image.")
        parser.print_help()
    else:
        # Here we can apply our functions (follow the struct of the readme architecture part)
        printArgs(args.image, args.pretrained_model, args.applyToForeground)
        # 2] Apply functions
        # Load the image
        image_path = args.image
        image = Image.open(image_path)

        # Convert the image to a list of lists (nested list) of pixels
        image_array = np.array(image)


#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # Apply deepmatting
        original_image_with_alpha = deep_matting(image_array)
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""        
       
        # Apply cycleGAN
        stylized_image = cycleGAN(image_array, args.pretrained_model)
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""        
        background_image = stylized_image

        # Ensure the images are of the same size
        if original_image_with_alpha.shape[:2] != background_image.shape[:2]:
            # Resize the background to match the foreground image
            background_image = resize(background_image, original_image_with_alpha.shape[:2], mode='constant', anti_aliasing=True)
            background_image = (background_image * 255).astype(np.uint8)  # Ensure that the background_image is in uint8
        
        # Superimpose the foreground image over the background
        # Convert both images to float to avoid data type issues during the computation
        foreground = original_image_with_alpha.astype(float)
        background = background_image.astype(float)
        
        # Normalize the alpha channel so that it is between 0 and 1
        alpha = foreground[:, :, 3] / 255.0
        
        # Perform the computation for the superimposition
        for color in range(0, 3):
            background[:, :, color] = alpha * foreground[:, :, color] + (1 - alpha) * background[:, :, color]
        
        # Convert the result back to uint8
        background = background.astype(np.uint8)
        
        # Save the resulting image
        Image.fromarray(background).save('image_superposee.png', 'PNG')



