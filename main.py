# import of our main modules
import deepMatting as dm
import cycleGAN as cg

# other imports
import argparse         # for arg
from PIL import Image   # for images
import matplotlib.pyplot as plt
import numpy as np

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

        # Apply deepmatting
        alpha_map = dm.deepmatting(image_array)

        # Apply cycleGAN
        stylized_image = cg.cycleGAN(image_array, args.pretrained_model)
        
        # Now matting:
        foreground = np.zeros(image_array.shape)
        background = np.zeros(image_array.shape)
        for i in range(len(image_array)):
            for j in range(len(image_array[0])):
                foreground[i][j] = [alpha_map[i][j]*image_array[i][j][c]/255 for c in [0,1,2]]
                background[i][j] = [(1-alpha_map[i][j])*stylized_image[i][j][c]/255 for c in [0,1,2]]

        # Combine components
        res = background + foreground
    
        # Display the original image
        plt.subplot(151)
        plt.imshow(image_array)
        plt.title("Original Image")
    
        # Display the alpha map
        plt.subplot(152)
        plt.imshow(alpha_map, cmap='gray')
        plt.title("Alpha Map")
    
        # Display the alpha part
        plt.subplot(153)
        plt.imshow(foreground)
        plt.title("foreground Part")
    
        # Display the stylized part
        plt.subplot(154)
        plt.imshow(background)
        plt.title("bg Part")
    
        # Display the result after matting
        plt.subplot(155)
        plt.imshow(res)
        plt.title("Result after Matting")
    
        plt.show()

