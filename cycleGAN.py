"""
Code linking CycleGAN repo and our project
"""
from cycleGAN.cycleGAN_bis import cycleGAN_bis 

def cycleGAN(image, pretrained_model):
    
    # to test
    '''
    res = []
    for ligne in image:
        nouvelle_ligne = []
        for pixel in ligne:
            nouvelle_ligne.append([pixel[0], 0, 0])
        res.append(nouvelle_ligne)
        '''
    return cycleGAN_bis(image, pretrained_model)
