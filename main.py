# import of our main modules
from deepMatting import deepmatting
from cycleGAN import cycleGAN

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


#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # Apply deepmatting
        original_image_with_alpha = deepmatting(image_array)
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""        
       
        # Apply cycleGAN
        stylized_image = cycleGAN(image_array, args.pretrained_model)
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""        
        background_image = stylized_image

        # Assurez-vous que les images ont la même taille
        if original_image_with_alpha.shape[:2] != background_image.shape[:2]:
            # Redimensionner le fond pour correspondre à l'image de premier plan
            from skimage.transform import resize
            background_image = resize(background_image, original_image_with_alpha.shape[:2], mode='constant', anti_aliasing=True)
            background_image = (background_image * 255).astype(np.uint8)  # Assurez-vous que background_image est en uint8

        # Superposer l'image de premier plan avec le fond
        # Convertir les deux images en float pour éviter des problèmes de type de données pendant le calcul
        foreground = original_image_with_alpha.astype(float)
        background = background_image.astype(float)

        # Normaliser l'alpha canal pour qu'il soit compris entre 0 et 1
        alpha = foreground[:, :, 3] / 255.0

        # Faire le calcul pour la superposition
        # Notez que le dernier canal de 'foreground' est l'alpha
        for color in range(0, 3):  # Vous parcourez seulement les trois premiers canaux (R, G, B)
            background[:, :, color] = alpha * foreground[:, :, color] + (1 - alpha) * background[:, :, color]

        # Convertir le résultat en uint8
        background = background.astype(np.uint8)

        # Sauvegarder l'image résultante
        Image.fromarray(background).save('image_superposee.png', 'PNG')

#plus nécessaire

          # Now matting:
        #foreground = np.zeros(image_array.shape)
        #background = np.zeros(image_array.shape)
        #for i in range(len(image_array)):
            #for j in range(len(image_array[0])):
                #foreground[i][j] = [alpha_map[i][j]*image_array[i][j][c]/255 for c in [0,1,2]]
                #background[i][j] = [(1-alpha_map[i][j])*stylized_image[i][j][c]/255 for c in [0,1,2]]
        #res = background + foreground


#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""        
'''
    
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
    
        plt.show()'''