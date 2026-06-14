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
    # Copions la pièce sans défaut pour lui ajouter des défauts
    img = generate_good_part().copy()
    # On ajoute une rayure sous forme de ligne, qui commence à la position (100,150) et finit à (300,250). Elle est plus foncé que
    # que le disque (30) et est remplie (-1)
    cv2.line(img, (100, 150), (300, 250), (30, 30, 30), 4)
    # un ajoute un cercle, représentant une tache. Elle est située à (280,140), a un rayon de 20, est gris foncé (40)
    # et est remplie (-1)
    cv2.circle(img, (280, 140), 20, (40, 40, 40), -1)
    # On ajoute une éllipse, qui représente un bout de la pièce manquante. Le (200,200,200) est important, car 
    # il donne à l'éllipse la même couleure que le fond de l'image, créant l'illusion de brisure
    cv2.ellipse(img, (200, 80), (25, 15), 45, 0, 180, (200, 200, 200), -1)
    return img


def detect_defects(img):
    """
    Détecte les défauts sur une pièce industrielle.
    Retourne : (image annotée, liste de défauts détectés)
    """
    defects = []
    annotated = img.copy()

    # l'image en couleur (BGR). On la met en différents tons de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Détection de la pièce principale ---
    # on floue légèrement l'image en appliquant une gaussienne pour élimier le bruit, donc ici les petits
    # détails inutiles. "gray" indique que l'image d'origine est en grayscale. (9,9) est la dimension matricielle
    # du floue. "2" indique l'intensité du floue 
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Comme notre pièce est composé de deux cercles, on utilise la fonction HoughCircles, qui détecte les formes
    # circulaires dans une image grayscale. "dp=1.2" indique la résolution de la recherche sera un peu plus basse que
    # la vrai résolution (1), mais ca rend le calcul plus rapide.
    # "miniDist=100" indique qu'il y a au moins 100 pixels de différence entre chaque cercle
    # "param1" indique la valeur du contraste nécéssaire pour qu'un pixel soit considéré comme un bord. On lui donne une valeur
    # de 50, car tout les bords sont bien définit dans notre image.
    # "param2" indique le seuil de certitude pour qu'une forme soit considéré comme un cercle.  "param2=30" indique
    # que 30 pixels de bords soient d'accord pour être considéré comme faisant partit d'un cercle.
    # minRadius et maxRadius indiquent l'intervalle de valeurs possibles pour les rayons des cercles qu'on veut détecter
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=100, param1=50, param2=30,
        minRadius=80, maxRadius=150
    )

    # Message si HoughCircles ne trouve pas de cercle
    if circles is None:
        defects.append("ERREUR : Pièce non détectée")
        return annotated, defects

    # Cercle principal détecté
    x, y, r = map(int, circles[0][0])
    cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
    cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

    # --- Détection des défauts par analyse de texture ---
    # Masque de la pièce seulement
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), r - 5, 255, -1)
    cv2.circle(mask, (x, y), 45, 0, -1)  # Exclure le trou central

    piece_region = cv2.bitwise_and(gray, gray, mask=mask)

    # Seuillage pour trouver les zones anormalement sombres (défauts)
    mean_val = cv2.mean(piece_region, mask=mask)[0]
    _, thresh = cv2.threshold(
        piece_region, mean_val * 0.75, 255, cv2.THRESH_BINARY_INV
    )
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Trouver les contours des défauts
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defect_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:  # Ignorer le bruit
            defect_count += 1
            x_d, y_d, w_d, h_d = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x_d, y_d),
                         (x_d + w_d, y_d + h_d), (0, 0, 255), 2)
            cv2.putText(annotated, f"Defaut {defect_count}",
                       (x_d, y_d - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 255), 1)

    if defect_count > 0:
        defects.append(f"{defect_count} défaut(s) détecté(s)")
    else:
        defects.append("Aucun défaut détecté")

    return annotated, defects
