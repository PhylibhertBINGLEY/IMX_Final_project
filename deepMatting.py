"""
code to interface deepmatting and our project
"""
""""""""""""""""""""""""""""""""""""""""""""""""
#AJOUT THOMAS 
import cv2
from ultralytics import YOLO
import cvzone
import tkinter as tk
from tkinter import filedialog,simpledialog
import os

def get_user_input():
    """ Affiche une boîte de dialogue pour obtenir une entrée de l'utilisateur """
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale
    user_input = simpledialog.askinteger("Input", "Voulez-vous extraire les humains ?\n0 pour oui, 1 pour non", parent=root)
    root.destroy()
    return user_input
def Image_Croppe():
    # Initialisation de Tkinter et cache de la fenêtre principale
    root = tk.Tk()
    root.withdraw()

    # Demande à l'utilisateur de sélectionner une image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    # Vérifie si un fichier a été sélectionné
    if file_path:
        # Vérifier l'extension du fichier
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            print(f"Image sélectionnée : {file_path}")
            
            # Initialisation du modèle YOLO à partir des poids pré-entraînés
            model = YOLO('yolov8n.pt')

            # Effectuer la détection sur l'image
            results = model(file_path)  # Remplacez par le chemin vers votre image
            print(len(results))
            value = get_user_input()
            
            if value is not None:
                print(f"Choix de l'utilisateur : {value}")

                image = cv2.imread(file_path)  # Remplacez par le chemin vers votre image
                print(value)
                for r in results:
                    boxes=r.boxes
                # Parcourir les nounding box pour les dessiner sur l'image
                    for bbox in boxes:
                        if value==0:
                            list_bbox=[]
                    # Extraire les informations de la boîte englobante
                            print(bbox[0].xyxy)
                            x1, y1, x2, y2=bbox[0].xyxy[0]
                        

                            # Obtenez les dimensions de l'image
                            height, width, _ = image.shape

                            # Définissez votre marge de sécurité
                            marge = 10  # par exemple, 10 pixels de marge sur chaque côté

                            # Agrandissez la bounding box en ajoutant/substrayant la marge aux coordonnées
                            x1 = max(x1 - marge, 0)  # Ne doit pas être inférieur à 0
                            y1 = max(y1 - marge, 0)  # Ne doit pas être inférieur à 0
                            x2 = min(x2 + marge, width - 1)  # Ne doit pas dépasser la largeur de l'image
                            y2 = min(y2 + marge, height - 1)  # Ne doit pas dépasser la hauteur de l'image

                             # x1, y1, x2, y2 ont été ajustés pour inclure la marge tout en restant à l'intérieur de l'image

                            x1, y1, x2, y2=int(x1), int(y1), int(x2), int(y2)
                            list_bbox.append((x1,x2,y1,y2))
                            x1,x2,y1,y2=list_bbox[0]
                            cropped_image = image[y1:y2, x1:x2]

                            # Afficher l'image recadrée
                            # cv2.imshow('Cropped Image', cropped_image)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            cv2.imwrite('cropped_image.jpg', cropped_image)

                            w, h = x2 - x1, y2 - y1

                            # Appliquer un seuil de confiance
                            if bbox[0].conf[0]> 0.50 and int(bbox[0].cls[0])==0 :
                                # Dessiner la bounding box
                                cvzone.cornerRect(image, (x1, y1, w, h))
                cv2.imshow('Image', image)
                # x1, y1, x2, y2 oordonnées de la bounding box
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Aucune entrée fournie par l'utilisateur.")
        else:
            print("Le fichier sélectionné n'est pas un fichier valide (JPG ou PNG).")
    else:
        print("Aucune image n'a été sélectionnée.")

    # Fermeture de l'interface graphique Tkinter
    root.destroy()
    if value==0:
        return(cropped_image)
    else :
        return(image)

Image_Croppe()
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
