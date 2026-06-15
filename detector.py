import cv2
import numpy as np


# Créons d'abord notre pièce sans défaut
def generate_good_part():
    """Génère une image d'une pièce sans défaut (cercle parfait)"""
    # D'abord on créer un image carré de 400x400 pixels. le "3" indique qu'elle est en couleur BGR.
    # Multiplier par 200 remplit l'image d'un gris clair. C'est notre fond gris 
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200  
    # La fonction circle dessine un cercle. Il est centré au point (200,200), donc au milieu. Il y un rayon de 120
    # Le (80, 80 , 80) donne sa couleur, un gris foncé. L'épaisseur -1 signifie que le cercle est entièrement rempli.
    cv2.circle(img, (200, 200), 120, (80, 80, 80), -1)    
    # puis on créer le trou au centre. Il est positionné au milieu (200,200), avec un rayon plus petit (40)
    # (150,150,150) Donne un gris plus claire, car 150 est plus proche de 255 (blanc) que 80. -1 indique que 
    # le cercle est remplit
    cv2.circle(img, (200, 200), 40, (150, 150, 150), -1) 
    return img

# créon maitenant notre pièce avec des défauts
def generate_defective_part():
    """Génère une image d'une pièce avec défauts (rayures et tache)"""
    img = generate_good_part().copy()
    
    # Rayure en deux segments qui s'arrêtent aux bords du trou (r=40)
    # Segment gauche : de x=100 jusqu'au bord gauche du trou
    cv2.line(img, (100, 150), (163, 210), (30, 30, 30), 4)
    # Segment droit : du bord droit du trou jusqu'à x=300
    cv2.line(img, (237, 230), (300, 260), (30, 30, 30), 4)
    
    # Tache
    cv2.circle(img, (280, 140), 20, (40, 40, 40), -1)
    
    # Ébréchure sur le bord
    cv2.ellipse(img, (200, 80), (25, 15), 45, 0, 180, (200, 200, 200), -1)
    
    return img


def detect_defects(img):
    defects = []
    annotated = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Détection de la pièce principale ---
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=100, param1=50, param2=30,
        minRadius=80, maxRadius=150
    )

    if circles is None:
        defects.append("ERREUR: Piece non detectee")
        return annotated, defects

    x, y, r = map(int, circles[0][0])
    cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
    cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

    # Masque de la pièce seulement (sans le trou central)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), r - 5, 255, -1)
    cv2.circle(mask, (x, y), 45, 0, -1)

    piece_region = cv2.bitwise_and(gray, gray, mask=mask)
    mean_val = cv2.mean(piece_region, mask=mask)[0]

    # --- Détection zones SOMBRES (rayures, taches) ---
    _, thresh_dark = cv2.threshold(
        piece_region, mean_val * 0.75, 255, cv2.THRESH_BINARY_INV
    )
    thresh_dark = cv2.bitwise_and(thresh_dark, thresh_dark, mask=mask)

    # --- Détection zones CLAIRES (brisures, éclats) ---
    _, thresh_light = cv2.threshold(
        piece_region, mean_val * 1.15, 255, cv2.THRESH_BINARY
    )
    thresh_light = cv2.bitwise_and(thresh_light, thresh_light, mask=mask)

    # Combiner les deux
    thresh_combined = cv2.bitwise_or(thresh_dark, thresh_light)

    # Dilatation plus petite pour fusionner sans effacer
    kernel_merge = np.ones((8, 8), np.uint8)
    thresh_combined = cv2.dilate(thresh_combined, kernel_merge, iterations=2)
    thresh_combined = cv2.erode(thresh_combined, kernel_merge, iterations=1)

    contours, _ = cv2.findContours(
        thresh_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defect_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            defect_count += 1
            x_d, y_d, w_d, h_d = cv2.boundingRect(cnt)
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