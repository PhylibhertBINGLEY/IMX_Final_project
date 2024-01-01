"""
code to interface deepmatting and our project
"""

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