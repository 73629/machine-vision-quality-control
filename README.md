# Machine Vision Quality Control Simulator

Par Nicolas Desnoyers

Simulateur de contrôle qualité par vision machine, développé en Python avec OpenCV.
Conçu pour démontrer les principes de base de l'inspection automatisée de pièces industrielles.

## Description

Ce projet simule un système de vision machine capable de détecter des défauts sur des
CD sur une ligne de production. Le système génère des pièces de test synthétiques et applique 
des algorithmes de traitement d'image pour identifier automatiquement les anomalies.

## Défauts détectés

- des rayures (les deux lignes noires)
- Une tache (le cercle noir)
- un bout brisé (la zone claire sur le bord)

## Technologies utilisées

- Python 3.x
- OpenCV (cv2) — traitement d'image et détection de contours
- NumPy — manipulation de matrices
- Matplotlib — visualisation des résultats

## Fonctions principales OpenCV utilisées

Pour mettre un flou gaussien sur l'image, on utilise :
$$GaussianBlur(src, dst, ksize, sigmaX)$$

**Avec :**
* $src$ : L'image d'entré
* $dst$ : L'image de sortie
* $ksize$ : La dimension de la matrice kernel qui floue l'image
* $sigmaX$ : L'écart-type en x de la gaussienne

Pour détecter les cercles dans l'image (transformée de Hough), on utilise :
$$HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)$$

**Avec :**
* $image$ : L'image d'entrée en niveaux de gris
* $method$ : La méthode de détection (HOUGH_GRADIENT)
* $dp$ : Le ratio de résolution entre l'image et l'accumulateur de Hough
* $minDist$ : La distance minimale entre les centres de cercles détectés
* $param1$ : Le seuil supérieur pour la détection des contours (Canny)
* $param2$ : Le seuil d'accumulation pour la détection des centres de cercles
* $minRadius$, $maxRadius$ : Les rayons minimal et maximal des cercles à détecter

---

Pour binariser l'image selon un seuil (segmentation), on utilise :
$$threshold(src, dst, thresh, maxval, type)$$

**Avec :**
* $src$ : L'image d'entrée en niveaux de gris
* $dst$ : L'image binarisée de sortie
* $thresh$ : La valeur de seuil
* $maxval$ : La valeur assignée aux pixels dépassant le seuil
* $type$ : Le type de seuillage (THRESH_BINARY ou THRESH_BINARY_INV)

---

Pour isoler une région d'intérêt à l'aide d'un masque, on utilise :
$$bitwise\__and(src1, src2, mask)$$

**Avec :**
* $src1$, $src2$ : Les images sur lesquelles appliquer l'opération
* $mask$ : Le masque binaire définissant la région d'intérêt

---

Pour combler les discontinuités d'une forme détectée (dilatation morphologique), on utilise :
$$dilate(src, dst, kernel, iterations)$$

**Avec :**
* $src$ : L'image binaire d'entrée
* $dst$ : L'image dilatée de sortie
* $kernel$ : La matrice structurante définissant la forme et la taille de la dilatation
* $iterations$ : Le nombre de fois où l'opération est appliquée

---

Pour réduire la dilatation excédentaire (érosion morphologique), on utilise :
$$erode(src, dst, kernel, iterations)$$

**Avec :**
* $src$ : L'image dilatée d'entrée
* $dst$ : L'image érodée de sortie
* $kernel$ : La matrice structurante définissant la forme et la taille de l'érosion
* $iterations$ : Le nombre de fois où l'opération est appliquée

---

Pour extraire les contours des défauts détectés, on utilise :
$$findContours(image, contours, hierarchy, mode, method)$$

**Avec :**
* $image$ : L'image binaire d'entrée
* $contours$ : La liste des contours détectés (sortie)
* $hierarchy$ : L'information de hiérarchie entre les contours (sortie)
* $mode$ : Le mode de récupération des contours (RETR_EXTERNAL)
* $method$ : La méthode d'approximation des contours (CHAIN_APPROX_SIMPLE)

## Installation

```bash
pip install opencv-python numpy matplotlib
```

## Utilisation

```bash
python main.py
```

Le programme génère automatiquement deux pièces de test (une sans défaut, une avec
défauts), analyse chacune et sauvegarde les résultats dans le dossier `images/`.

## Algorithme de détection

1. Détection de la pièce principale par la fonction HoughCircles
2. Création d'un masque excluant le trou central et le fond
3. Calcul de la valeur moyenne de gris sur la surface de la pièce
4. Seuillage adaptatif pour détecter les zones anormalement sombres (rayures, taches)
   et claires (ébréchures)
5. Dilatation morphologique pour fusionner les contours proches
6. Classification : ACCEPTEE si aucun défaut, REJETEE sinon

## Résultats

![Résultats](images/quality_control_results.png)

## Auteur

Nicolas Desnoyers — Étudiant en génie physique, Polytechnique Montréal  
[GitHub](https://github.com/73629)