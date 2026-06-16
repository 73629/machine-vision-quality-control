import cv2
import numpy as np


# Créons d'abord notre pièce sans défaut
def generate_good_part():
    """Génère une image d'une pièce sans défaut (cercle parfait)"""
    # D'abord on créer un image carré de 400x400 pixels. le "3" indique qu'elle est en couleur BGR.
    # Multiplier par 200 remplit l'image d'un gris clair. C'est notre fond gris 
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200  
    # La fonction circle dessine un cercle. Il est centré au point (200,200), donc au milieu. 
    # Il y un rayon de 120
    # Le (80, 80 , 80) donne sa couleur, un gris foncé. 
    # L'épaisseur -1 signifie que le cercle est entièrement rempli.
    cv2.circle(img, (200, 200), 120, (80, 80, 80), -1)    
    # On créer le trou au centre. Il est centré en (200,200)
    # Il a un rayon de 40
    # (150,150,150) Donne un gris plus claire, car 150 est plus proche de 255 (blanc) que 80.
    # -1 indique que le cercle est remplit
    cv2.circle(img, (200, 200), 40, (150, 150, 150), -1) 
    return img

# créons maitenant notre pièce avec des défauts
def generate_defective_part():
    """Génère une image d'une pièce avec défauts (rayures et tache)"""
    img = generate_good_part().copy()
    
    # On créer la première rayure
    # Elle commence en (100,150) et finit à (163,250)
    cv2.line(img, (100, 150), (163, 210), (30, 30, 30), 4)
    # On créer la deuxième rayure
    # Elle commence en (237,230) et finit en (300,260)
    cv2.line(img, (237, 230), (300, 260), (30, 30, 30), 4)
    
    # On créer la tache. Elle a un rayon de 20, 40 lui donne du gris très foncé et -1 la remplie
    cv2.circle(img, (280, 140), 20, (40, 40, 40), -1)
    
    # On créer le bord brisé
    cv2.ellipse(img, (200, 80), (25, 15), 45, 0, 180, (200, 200, 200), -1)
    
    return img

# créon maitenant notre fonction de détection de défauts
def detect_defects(img):
    defects = []
    annotated = img.copy()

    # l'image en initiallement en couleur (BGR). On la met en grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # On applique un flou gaussien pour éliminer les petits détails
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # On détecte les cercles dans notre image grayscaled floutée
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=100, param1=50, param2=30,
        minRadius=80, maxRadius=150
    )
    # Si HoughCircle de trouve pas de cercle :
    if circles is None:
        defects.append("ERREUR: Piece non detectee")
        return annotated, defects
    
    # On sauvergarde les cordonées (x,y) et le rayon r du premier cercle détecté (le principal)
    x, y, r = map(int, circles[0][0])
    # On dessine les cercles verts sur ce cercle
    cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
    cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

    # On créer le masque, donc notre pièce complètement blanche dans un fond noir avec un rayon légèrement plus petit (r-5)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), r - 5, 255, -1)
    cv2.circle(mask, (x, y), 45, 0, -1)
    
    # On colle la pièce sur le masque. Le blanc devient l'image grayscaled, et le noir reste noire
    piece_region = cv2.bitwise_and(gray, gray, mask=mask)
    # On calcule la valeur grise moyenne du CD
    mean_val = cv2.mean(piece_region, mask=mask)[0]

    # On détecte les zones qui sont 75% plus sombre qe la valeur gris moyenne, ce sont les défauts sombres.
    # puis on les mets en blanc sur un fond noir
    _, thresh_dark = cv2.threshold(
        piece_region, mean_val * 0.75, 255, cv2.THRESH_BINARY_INV
    )
    thresh_dark = cv2.bitwise_and(thresh_dark, thresh_dark, mask=mask)

    # On détecte les zones qui sont 15% plus claire que la valeur grise moyenne, ce sont les défauts claires (brisure).
    # puis on les mets en blanc dans un fond noir
    _, thresh_light = cv2.threshold(
        piece_region, mean_val * 1.15, 255, cv2.THRESH_BINARY
    )
    thresh_light = cv2.bitwise_and(thresh_light, thresh_light, mask=mask)

    # On combine les deux pour créer une image contenant tout les défauts en blanc sur un fond noir.
    thresh_combined = cv2.bitwise_or(thresh_dark, thresh_light)

    kernel_merge = np.ones((8, 8), np.uint8)
    # On dilate deux fois les morceaux blancs pour que les petits morceaux se fusionnent en 1 gros défaut
    thresh_combined = cv2.dilate(thresh_combined, kernel_merge, iterations=2)
    # Puis on diminue les défauts fusionnés une fois, pour qu'ils soient un peu plus gros que leur taille de départ.
    thresh_combined = cv2.erode(thresh_combined, kernel_merge, iterations=1)
 
    # On sauvegarde les contours des défauts
    contours, _ = cv2.findContours(
        thresh_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defect_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # On vérifie que l'air de chaque défaut est assez grand pour compter comme un défaut
        if area > 100:
            defect_count += 1
            x_d, y_d, w_d, h_d = cv2.boundingRect(cnt)
            # On trace un rectangle rouge autour du défaut
            cv2.rectangle(annotated, (x_d, y_d),
                         (x_d + w_d, y_d + h_d), (0, 0, 255), 2)
            cv2.putText(annotated, f"Defaut {defect_count}",
                       (x_d, y_d - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 255), 1)

    if defect_count > 0:
        defects.append(f"{defect_count} defaut(s) detecte(s)")
    else:
        defects.append("Aucun defaut detecte")

    return annotated, defects