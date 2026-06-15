import cv2
import numpy as np
import matplotlib.pyplot as plt
from detector import generate_good_part, generate_defective_part, detect_defects

def analyze_part(img, label):
    """Analyse une pièce et retourne l'image annotée avec résultats"""
    annotated, defects = detect_defects(img)

    verdict = "REJETEE" if any("defaut(s)" in d for d in defects) else "ACCEPTEE"
    
    # Couleurs distinctes et lisibles
    color_verdict = (0, 180, 0) if verdict == "ACCEPTEE" else (0, 0, 255)  # Vert foncé ou Rouge
    color_details = (0, 120, 0) if verdict == "ACCEPTEE" else (0, 0, 200)  # Vert foncé ou Rouge foncé

    cv2.putText(annotated, f"{label}: {verdict}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color_verdict, 2)

    for i, defect in enumerate(defects):
        cv2.putText(annotated, defect,
                    (10, 60 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color_details, 1)

    return annotated, verdict
# === PROGRAMME PRINCIPAL ===
print("=== Système de contrôle qualité par vision machine ===\n")

# Générer les pièces
good = generate_good_part()
defective = generate_defective_part()

# Sauvegarder les images originales
cv2.imwrite("images/good_part.png", good)
cv2.imwrite("images/defective_part.png", defective)

# Analyser les deux pièces
result_good, verdict_good = analyze_part(good, "Piece 1")
result_defective, verdict_defective = analyze_part(defective, "Piece 2")

# Sauvegarder les résultats
cv2.imwrite("images/result_good.png", result_good)
cv2.imwrite("images/result_defective.png", result_defective)

# Affichage matplotlib
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Système de contrôle qualité par vision machine", fontsize=14)

axes[0][0].imshow(cv2.cvtColor(good, cv2.COLOR_BGR2RGB))
axes[0][0].set_title("Pièce 1 – Originale")
axes[0][0].axis("off")

axes[0][1].imshow(cv2.cvtColor(result_good, cv2.COLOR_BGR2RGB))
axes[0][1].set_title(f"Pièce 1 – Analyse : {verdict_good}")
axes[0][1].axis("off")

axes[1][0].imshow(cv2.cvtColor(defective, cv2.COLOR_BGR2RGB))
axes[1][0].set_title("Pièce 2 – Originale")
axes[1][0].axis("off")

axes[1][1].imshow(cv2.cvtColor(result_defective, cv2.COLOR_BGR2RGB))
axes[1][1].set_title(f"Pièce 2 – Analyse : {verdict_defective}")
axes[1][1].axis("off")

plt.tight_layout()
plt.savefig("images/quality_control_results.png", dpi=150)
plt.show()

print(f"Pièce 1 : {verdict_good}")
print(f"Pièce 2 : {verdict_defective}")
print("\nRésultats sauvegardés dans images/")