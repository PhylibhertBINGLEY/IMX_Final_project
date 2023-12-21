# import of our main modules
import deepMatting as dm
import cycleGAN as cg

# other imports
import argparse      # for arg

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
