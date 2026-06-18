import cv2
import numpy as np


# Créons d'abord le CD sans défauts
def generate_good_part():
    
    # D'abord on crée un image carrée de 400x400 pixels. le "3" indique qu'elle est en couleur BGR.
    # Multiplier par 200 remplit l'image d'un gris clair. C'est notre fond gris 
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200  
    # La fonction circle dessine un cercle. Il est centré au point (200,200), donc au milieu. 
    # Il y un rayon de 120
    # Le (80, 80 , 80) donne sa couleur, un gris foncé. 
    # L'épaisseur -1 signifie que le cercle est entièrement rempli.
    cv2.circle(img, (200, 200), 120, (80, 80, 80), -1)    
    # On crée le trou au centre. Il est centré en (200,200)
    # Il a un rayon de 40
    # (150,150,150) Donne un gris plus clair, car 150 est plus proche de 255 (blanc) que 80.
    # -1 indique que le cercle est rempli
    cv2.circle(img, (200, 200), 40, (150, 150, 150), -1) 
    return img

# Créons maintenant le CD avec des défauts
def generate_defective_part():
    img = generate_good_part().copy()
    
    # On crée la première rayure
    # Elle commence en (100,150) et finit à (163,250)
    cv2.line(img, (100, 150), (163, 210), (30, 30, 30), 4)
    # On crée la deuxième rayure
    # Elle commence en (237,230) et finit en (300,260)
    cv2.line(img, (237, 230), (300, 260), (30, 30, 30), 4)
    
    # On créer la tache. Elle a un rayon de 20, 40 lui donne du gris très foncé et -1 la remplit
    cv2.circle(img, (280, 140), 20, (40, 40, 40), -1)
    
    # On crée le bord brisé
    cv2.ellipse(img, (200, 80), (25, 15), 45, 0, 180, (200, 200, 200), -1)
    
    return img

# Créons maitenant notre fonction de détection de défauts
def detect_defects(img):
    defects = []
    annotated = img.copy()

    # L'image est initialement en couleur (BGR). On la met en grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # On applique un flou gaussien pour éliminer les petits détails
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # On détecte les cercles dans notre image grayscaled floutée.
    # "dp=1.2" indique que la résolution de la recherche sera un peu plus basse que la vraie résolution (1), mais ça rend le calcul plus rapide
    # "miniDist=100" indique qu'il y a au moins 100 pixels de différence entre chaque cercle
    # "param1=50" indique qu'il ne faut pas un grand contraste pour qu'un pixel soit considéré comme un bord
    # "param2=30" indique qu'il faut que 30 pixels de bord soient "d'accord" pour être considéré comme faisant partie d'un cercle
    # minRadius et maxRadius indiquent l'intervalle de valeurs possibles pour les rayons des cercles qu'on veut détecter
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=100, param1=50, param2=30,
        minRadius=80, maxRadius=150
    )
    # Si HoughCircle ne trouve pas de cercle :
    if circles is None:
        defects.append("ERREUR: Pièce non détectée")
        return annotated, defects
    
    # On sauvegarde les coordonnées (x,y) et le rayon r du premier cercle détecté (ici le principal)
    x, y, r = map(int, circles[0][0])
    # On dessine les cercles verts sur ce cercle
    cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
    cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

    # On crée le masque, donc la forme du CD remplie de blanc dans un fond noir avec un rayon légèrement plus petit (r-5)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), r - 5, 255, -1)
    cv2.circle(mask, (x, y), 45, 0, -1)
    
    # On colle le CD sur le masque. Le blanc devient l'image grayscaled, et le noir reste noir
    piece_region = cv2.bitwise_and(gray, gray, mask=mask)
    # On calcule la valeur grise moyenne du CD
    mean_val = cv2.mean(piece_region, mask=mask)[0]

    # On détecte les zones qui sont 25% plus sombre que la valeur grise moyenne, ce sont les défauts sombres.
    # Puis on les met en blanc sur un fond noir
    _, thresh_dark = cv2.threshold(
        piece_region, mean_val * 0.75, 255, cv2.THRESH_BINARY_INV
    )
    thresh_dark = cv2.bitwise_and(thresh_dark, thresh_dark, mask=mask)

    # On détecte les zones qui sont 15% plus claire que la valeur grise moyenne, ce sont les défauts clairs (brisure).
    # Puis on les met en blanc dans un fond noir
    _, thresh_light = cv2.threshold(
        piece_region, mean_val * 1.15, 255, cv2.THRESH_BINARY
    )
    thresh_light = cv2.bitwise_and(thresh_light, thresh_light, mask=mask)

    # On combine les deux pour créer une image contenant tous les défauts en blanc sur un fond noir.
    thresh_combined = cv2.bitwise_or(thresh_dark, thresh_light)
 
    # Pour s'assurer que plusieurs petits morceaux d'un défaut ne soient pas comptés comme plusieurs défauts, on
    # dilate tous les morceaux blancs pour fusionner ces petits défauts potentiels, puis on les réduit. Comme on
    # dilate deux fois mais on ne réduit qu'une fois, les défauts seront un peu plus gros que leur taille de départ.
    kernel_merge = np.ones((8, 8), np.uint8)
    thresh_combined = cv2.dilate(thresh_combined, kernel_merge, iterations=2)
    thresh_combined = cv2.erode(thresh_combined, kernel_merge, iterations=1)
 
    # On sauvegarde les contours des défauts
    contours, _ = cv2.findContours(
        thresh_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defect_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # On vérifie que l'aire de chaque défaut est assez grand pour compter comme un défaut
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