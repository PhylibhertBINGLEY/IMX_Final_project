"""
code to interface deepmatting and our project
"""
""""""""""""""""""""""""""""""""""""""""""""""""
#AJOUT THOMAS 
import cv2
from ultralytics import YOLO
import cvzone
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Add
import numpy as np
from glob import glob
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm



def get_user_input():
    """ Demande à l'utilisateur s'il veut extraire les humains """
    while True:
        user_input = input("Do you want to extract humans? Enter 0 for yes, 1 for no: ")
        if user_input in ['0', '1']:
            return int(user_input)
        print("Invalid input, please enter 0 or 1.")

def image_crop():
    import os
    file_path = '/content/1.jpg' # Demande à l'utilisateur de saisir le chemin du fichier

    # Vérifie si un fichier a été sélectionné
    if file_path:
        # Vérifie l'extension du fichier
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            print(f"Selected image: {file_path}")

            image_save = cv2.imread(file_path)
            # Initialise le modèle YOLO à partir des poids pré-entraînés
            model = YOLO('yolov8n.pt')

            # Effectue la détection sur l'image
            results = model(file_path)
            print(len(results))
            value = get_user_input()

            if value is not None:
                print(f"User choice: {value}")

                image = cv2.imread(file_path)
                print(value)
                for r in results:
                    boxes = r.boxes
                    # Parcourt les bounding boxes pour les dessiner sur l'image
                    for bbox in boxes:
                        if value == 0:
                            list_bbox = []
                            print(bbox[0].xyxy)
                            x1, y1, x2, y2 = bbox[0].xyxy[0]
                        
                            # Obtient les dimensions de l'image
                            height, width, _ = image.shape

                            # Définit une marge de sécurité
                            margin = 10  # Par exemple, 10 pixels de marge de chaque côté

                            # Agrandit la bounding box en ajoutant/soustrayant la marge aux coordonnées
                            x1 = max(x1 - margin, 0)
                            y1 = max(y1 - margin, 0)
                            x2 = min(x2 + margin, width - 1)
                            y2 = min(y2 + margin, height - 1)

                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            list_bbox.append((x1, x2, y1, y2))
                            x1, x2, y1, y2 = list_bbox[0]
                            cropped_image = image[y1:y2, x1:x2]

                            # Chemin du dossier où vous voulez enregistrer l'image
                            folder_path = '/content/test'

                            # Vérifier si le dossier existe
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)
                            file_path = os.path.join(folder_path, '1.jpg')

                            # Enregistrer l'image
                            cv2.imwrite(file_path, cropped_image)

                            crop_copy = cropped_image.copy()

                            w, h = x2 - x1, y2 - y1

                            if bbox[0].conf[0] > 0.50 and int(bbox[0].cls[0]) == 0:
                                cvzone.cornerRect(image, (x1, y1, w, h))
                cv2.imshow('Image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No input provided by the user.")
        else:
            print("The selected file is not a valid file (JPG or PNG).")
    else:
        print("No image has been selected.")

    if value == 0:
        return (((x1, y1, w, h), image_save, crop_copy,True))
    else:
        return (((x1, y1, w, h), image_save, image_save,False))

# image_crop()
(x, y, w, h), Original_Im, Im_Crop,Check = image_crop()
# if Check:
#     Original_Im[y:y+h, x:x+w] = Im_Crop




def conv_block(inputs, out_ch, rate=1):
    x = Conv2D(out_ch, 3, padding="same", dilation_rate=1)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def RSU_L(inputs, out_ch, int_ch, num_layers, rate=2):
    """ Initial Conv """
    x = conv_block(inputs, out_ch)
    init_feats = x

    """ Encoder """
    skip = []
    x = conv_block(x, int_ch)
    skip.append(x)

    for i in range(num_layers-2):
        x = MaxPool2D((2, 2))(x)
        x = conv_block(x, int_ch)
        skip.append(x)

    """ Bridge """
    x = conv_block(x, int_ch, rate=rate)

    """ Decoder """
    skip.reverse()

    x = Concatenate()([x, skip[0]])
    x = conv_block(x, int_ch)

    for i in range(num_layers-3):
        x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = Concatenate()([x, skip[i+1]])
        x = conv_block(x, int_ch)

    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Concatenate()([x, skip[-1]])
    x = conv_block(x, out_ch)

    """ Add """
    x = Add()([x, init_feats])
    return x

def RSU_4F(inputs, out_ch, int_ch):
    """ Initial Conv """
    x0 = conv_block(inputs, out_ch, rate=1)

    """ Encoder """
    x1 = conv_block(x0, int_ch, rate=1)
    x2 = conv_block(x1, int_ch, rate=2)
    x3 = conv_block(x2, int_ch, rate=4)

    """ Bridge """
    x4 = conv_block(x3, int_ch, rate=8)

    """ Decoder """
    x = Concatenate()([x4, x3])
    x = conv_block(x, int_ch, rate=4)

    x = Concatenate()([x, x2])
    x = conv_block(x, int_ch, rate=2)

    x = Concatenate()([x, x1])
    x = conv_block(x, out_ch, rate=1)

    """ Addition """
    x = Add()([x, x0])
    return x

def u2net(input_shape, out_ch, int_ch, num_classes=1):
    """ Input Layer """
    inputs = Input(input_shape)
    s0 = inputs

    """ Encoder """
    s1 = RSU_L(s0, out_ch[0], int_ch[0], 7)
    p1 = MaxPool2D((2, 2))(s1)

    s2 = RSU_L(p1, out_ch[1], int_ch[1], 6)
    p2 = MaxPool2D((2, 2))(s2)

    s3 = RSU_L(p2, out_ch[2], int_ch[2], 5)
    p3 = MaxPool2D((2, 2))(s3)

    s4 = RSU_L(p3, out_ch[3], int_ch[3], 4)
    p4 = MaxPool2D((2, 2))(s4)

    s5 = RSU_4F(p4, out_ch[4], int_ch[4])
    p5 = MaxPool2D((2, 2))(s5)

    """ Bridge """
    b1 = RSU_4F(p5, out_ch[5], int_ch[5])
    b2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(b1)

    """ Decoder """
    d1 = Concatenate()([b2, s5])
    d1 = RSU_4F(d1, out_ch[6], int_ch[6])
    u1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d1)

    d2 = Concatenate()([u1, s4])
    d2 = RSU_L(d2, out_ch[7], int_ch[7], 4)
    u2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d2)

    d3 = Concatenate()([u2, s3])
    d3 = RSU_L(d3, out_ch[8], int_ch[8], 5)
    u3 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d3)

    d4 = Concatenate()([u3, s2])
    d4 = RSU_L(d4, out_ch[9], int_ch[9], 6)
    u4 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d4)

    d5 = Concatenate()([u4, s1])
    d5 = RSU_L(d5, out_ch[10], int_ch[10], 7)

    """ Side Outputs """
    y1 = Conv2D(num_classes, 3, padding="same")(d5)

    y2 = Conv2D(num_classes, 3, padding="same")(d4)
    y2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(y2)

    y3 = Conv2D(num_classes, 3, padding="same")(d3)
    y3 = UpSampling2D(size=(4, 4), interpolation="bilinear")(y3)

    y4 = Conv2D(num_classes, 3, padding="same")(d2)
    y4 = UpSampling2D(size=(8, 8), interpolation="bilinear")(y4)

    y5 = Conv2D(num_classes, 3, padding="same")(d1)
    y5 = UpSampling2D(size=(16, 16), interpolation="bilinear")(y5)

    y6 = Conv2D(num_classes, 3, padding="same")(b1)
    y6 = UpSampling2D(size=(32, 32), interpolation="bilinear")(y6)

    y0 = Concatenate()([y1, y2, y3, y4, y5, y6])
    y0 = Conv2D(num_classes, 3, padding="same")(y0)

    y0 = Activation("sigmoid", name="y0")(y0)
    y1 = Activation("sigmoid", name="y1")(y1)
    y2 = Activation("sigmoid", name="y2")(y2)
    y3 = Activation("sigmoid", name="y3")(y3)
    y4 = Activation("sigmoid", name="y4")(y4)
    y5 = Activation("sigmoid", name="y5")(y5)
    y6 = Activation("sigmoid", name="y6")(y6)

    model = tf.keras.models.Model(inputs, outputs=[y0, y1, y2, y3, y4, y5, y6])
    return model


def build_u2net_lite(input_shape, num_classes=1):
    out_ch = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    int_ch = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    model = u2net(input_shape, out_ch, int_ch, num_classes=num_classes)
    return model

#from model import build_u2net_lite, build_u2net

""" Global parameters """
H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    train_x = sorted(glob(os.path.join(path, "train", "blurred_image", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "validation", "P3M-500-NP", "original_image", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "validation", "P3M-500-NP", "mask", "*.png")))

    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds




# def training():
#     """ Seeding """
#     np.random.seed(42)
#     tf.random.set_seed(42)

#     """ Directory for storing files """
#     create_dir("files")

#     """ Hyperparameters """
#     batch_size = 4
#     lr = 1e-4
#     num_epochs = 1
#     model_path = os.path.join("files", "model.h5")
#     csv_path = os.path.join("files", "log.csv")

#     """ Dataset """
#     dataset_path = "/content/P3M-10k"
#     (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path)

#     print(f"Train: {len(train_x)} - {len(train_y)}")
#     print(f"Valid: {len(valid_x)} - {len(valid_y)}")

#     train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
#     valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

#     """ Model """
#     model = build_u2net_lite((H, W, 3))
#     model.load_weights(model_path)
#     model.compile(loss="binary_crossentropy", optimizer=Adam(lr))

#     callbacks = [
#         ModelCheckpoint(model_path, verbose=1, save_best_only=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
#         CSVLogger(csv_path),
#         EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
#     ]

#     model.fit(
#         train_dataset,
#         epochs=num_epochs,
#         validation_data=valid_dataset,
#         callbacks=callbacks
#     )



def main_matting():
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    for item in ["joint", "mask"]:
        create_dir(f"results/{item}")

    """ Load the model """
    model_path = '/content/files/model.h5'
    model = tf.keras.models.load_model(model_path)

    """ Dataset """
    images = glob("/content/test/*")
    print(f"Images: {len(images)}")

    """ Prediction """
    for x in tqdm(images, total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        pred = model.predict(x, verbose=0)

        line = np.ones((H, 10, 3)) * 255

        """ Joint and save mask """
        pred_list = []
        for item in pred:
            p = item[0] * 255
            p = np.concatenate([p, p, p], axis=-1)

            pred_list.append(p)
            pred_list.append(line)

        save_image_path = os.path.join("results", "mask", name)
        cat_images = np.concatenate(pred_list, axis=1)
        cv2.imwrite(save_image_path, cat_images)

        """ Save final mask """
        image_h, image_w, _ = image.shape

        y0 = pred[0][0]
        y0 = cv2.resize(y0, (image_w, image_h))
        y0 = np.expand_dims(y0, axis=-1)
        y0 = np.concatenate([y0, y0, y0], axis=-1)
        

        # # Extrait le premier plan en multipliant pixel par pixel avec le masque alpha
        foreground = (y0 * image).astype(np.uint8)

        # Affichez le premier plan
        plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Enlever les axes pour une meilleure visualisation
        plt.show()

        inv=1-y0
        back = (image*inv).astype(np.uint8)

        # Affichez le premier plan
        plt.imshow(cv2.cvtColor(back, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Enlever les axes pour une meilleure visualisation
        plt.show()

        # Affichez la shape de l'image y0
        print('Shape de y0:', foreground.shape)
        line = line = np.ones((image_h, 10, 3)) * 255

        cat_images = np.concatenate([image, line, y0*255, line, image*y0], axis=1)
        save_image_path = os.path.join("results", "joint", name)
        cv2.imwrite(save_image_path, cat_images)
        return((foreground,back))
# training()
main_matting()

""""""""""""""""""""""""""""""""""""""""""""""""
def deepmatting(image):
    
    # to test
    alpha = []
    for i,ligne in enumerate(image):
        nouvelle_ligne = []
        for pixel in ligne:
            if(i<len(image)/2):
                nouvelle_ligne.append(1)
            else:
                nouvelle_ligne.append(0)
        alpha.append(nouvelle_ligne)
    return alpha
