"""
Genera 12 matrices de confusión (few-shot y zero-shot) como archivos PDF,
con las etiquetas traducidas al español y los valores ya "hardcodeados"
(tomados de las imágenes originales).

Requisitos: matplotlib, numpy
Uso: python3 generar_matrices.py
Salida: un archivo .pdf por cada matriz en la carpeta ./salidas_pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Carpeta de salida
OUTPUT_DIR = "salidas_pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def graficar_matriz_confusion(matriz, etiquetas, cmap, nombre_archivo, titulo=None,
                               vmin=0, vmax=None):
    """
    Dibuja una matriz de confusión estilo sklearn (ConfusionMatrixDisplay)
    y la guarda como PDF.

    matriz: lista de listas (o array) con los conteos, filas = etiqueta real,
            columnas = etiqueta predicha.
    etiquetas: lista de nombres de clase, en el mismo orden que filas/columnas.
    cmap: colormap de matplotlib (p.ej. "Blues" o "Reds").
    nombre_archivo: nombre del PDF de salida (sin ruta).
    vmin, vmax: límites de la escala de color. Si vmax es None, se usa el
                máximo de la propia matriz (comportamiento independiente).
                Para comparar dos matrices (few-shot vs zero-shot), pasa el
                mismo vmax a ambas llamadas.
    """
    matriz = np.array(matriz)
    n = len(etiquetas)

    if vmax is None:
        vmax = matriz.max() if matriz.max() > 0 else 1

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(matriz, cmap=cmap, vmin=vmin, vmax=vmax)

    # Barra de color (misma escala si vmax se comparte entre matrices)
    fig.colorbar(im, ax=ax)

    # Ejes
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(etiquetas)
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")

    if titulo:
        ax.set_title(titulo)

    # Umbral de color para decidir el color del texto (blanco/negro)
    umbral = vmin + (vmax - vmin) / 2.0

    for i in range(n):
        for j in range(n):
            valor = matriz[i, j]
            color_texto = "white" if valor > umbral else "black"
            ax.text(j, i, f"{valor}", ha="center", va="center", color=color_texto)

    fig.tight_layout()
    ruta_salida = os.path.join(OUTPUT_DIR, nombre_archivo)
    fig.savefig(ruta_salida, format="pdf")
    plt.close(fig)
    print(f"Guardado: {ruta_salida}")


# ---------------------------------------------------------------------------
# 1) ABUSIVE - Few-shot (azul) y Zero-shot (rojo)
# ---------------------------------------------------------------------------
etiquetas_abusive = ["Bajo", "Alto"]

abusive_few_shot = [
    [2924, 5273],
    [545, 1620],
]

abusive_zero_shot = [
    [8197, 0],
    [2165, 0],
]

# ---------------------------------------------------------------------------
# 2) EMOTION - Few-shot y Zero-shot
# ---------------------------------------------------------------------------
etiquetas_emotion = ["Miedo", "Alegría", "Ira", "Tristeza", "Neutral", "Sorpresa"]

emotion_few_shot = [
    [2416, 0, 134, 4719, 0, 3],
    [134, 0, 9, 374, 0, 2],
    [418, 0, 27, 691, 0, 1],
    [181, 0, 14, 326, 0, 0],
    [143, 0, 7, 289, 0, 0],
    [161, 0, 10, 303, 0, 0],
]

emotion_zero_shot = [
    [432, 0, 0, 6146, 2, 0],
    [28, 0, 0, 455, 0, 0],
    [47, 0, 0, 996, 0, 0],
    [15, 0, 0, 467, 0, 0],
    [2, 0, 0, 386, 0, 0],
    [8, 0, 0, 434, 2, 0],
]

# ---------------------------------------------------------------------------
# 3) MFD (Moral Foundations Dictionary) - Few-shot y Zero-shot
# ---------------------------------------------------------------------------
etiquetas_mfd = ["Igual", "Vicio", "Virtud"]

mfd_few_shot = [
    [3428, 721, 2757],
    [485, 144, 465],
    [1217, 223, 922],
]

mfd_zero_shot = [
    [466, 0, 6440],
    [57, 0, 1037],
    [195, 0, 2167],
]

# ---------------------------------------------------------------------------
# 4) POLARIZATION - Few-shot y Zero-shot
# ---------------------------------------------------------------------------
etiquetas_polarization = ["Bajo", "Alto"]

polarization_few_shot = [
    [1457, 5055],
    [698, 3152],
]

polarization_zero_shot = [
    [6512, 0],
    [3850, 0],
]

# ---------------------------------------------------------------------------
# 5) SENTIMENT - Few-shot y Zero-shot
# ---------------------------------------------------------------------------
etiquetas_sentiment = ["Negativo", "Neutral", "Positivo"]

sentiment_few_shot = [
    [2876, 2783, 15],
    [1303, 2116, 16],
    [407, 834, 12],
]

sentiment_zero_shot = [
    [1067, 4584, 23],
    [440, 2965, 30],
    [136, 1092, 25],
]

# ---------------------------------------------------------------------------
# 6) VALENCE - Few-shot y Zero-shot
# ---------------------------------------------------------------------------
etiquetas_valence = ["Igual", "Negativo", "Positivo"]

valence_few_shot = [
    [120, 2227, 1326],
    [79, 1659, 916],
    [126, 2125, 1784],
]

valence_zero_shot = [
    [0, 2050, 1623],
    [0, 1584, 1070],
    [0, 1932, 2103],
]

# ---------------------------------------------------------------------------
# Generar todos los PDFs
# Cada par few-shot / zero-shot comparte la misma escala de color (mismo
# vmax = máximo entre ambas matrices), para que sean directamente comparables.
# ---------------------------------------------------------------------------
def vmax_compartido(*matrices):
    """Devuelve el valor máximo entre varias matrices, para usar como vmax común."""
    return max(np.array(m).max() for m in matrices)


if __name__ == "__main__":

    # --- Abusive ---
    vmax_abusive = vmax_compartido(abusive_few_shot, abusive_zero_shot)
    graficar_matriz_confusion(
        abusive_few_shot, etiquetas_abusive, "Blues", "abusive_few_shot.pdf",
        vmax=vmax_abusive,
    )
    graficar_matriz_confusion(
        abusive_zero_shot, etiquetas_abusive, "Reds", "abusive_zero_shot.pdf",
        vmax=vmax_abusive,
    )

    # --- Emotion ---
    vmax_emotion = vmax_compartido(emotion_few_shot, emotion_zero_shot)
    graficar_matriz_confusion(
        emotion_few_shot, etiquetas_emotion, "Blues", "emotion_few_shot.pdf",
        vmax=vmax_emotion,
    )
    graficar_matriz_confusion(
        emotion_zero_shot, etiquetas_emotion, "Reds", "emotion_zero_shot.pdf",
        vmax=vmax_emotion,
    )

    # --- MFD ---
    vmax_mfd = vmax_compartido(mfd_few_shot, mfd_zero_shot)
    graficar_matriz_confusion(
        mfd_few_shot, etiquetas_mfd, "Blues", "mfd_few_shot.pdf",
        vmax=vmax_mfd,
    )
    graficar_matriz_confusion(
        mfd_zero_shot, etiquetas_mfd, "Reds", "mfd_zero_shot.pdf",
        vmax=vmax_mfd,
    )

    # --- Polarization ---
    vmax_polarization = vmax_compartido(polarization_few_shot, polarization_zero_shot)
    graficar_matriz_confusion(
        polarization_few_shot, etiquetas_polarization, "Blues", "polarization_few_shot.pdf",
        vmax=vmax_polarization,
    )
    graficar_matriz_confusion(
        polarization_zero_shot, etiquetas_polarization, "Reds", "polarization_zero_shot.pdf",
        vmax=vmax_polarization,
    )

    # --- Sentiment ---
    vmax_sentiment = vmax_compartido(sentiment_few_shot, sentiment_zero_shot)
    graficar_matriz_confusion(
        sentiment_few_shot, etiquetas_sentiment, "Blues", "sentiment_few_shot.pdf",
        vmax=vmax_sentiment,
    )
    graficar_matriz_confusion(
        sentiment_zero_shot, etiquetas_sentiment, "Reds", "sentiment_zero_shot.pdf",
        vmax=vmax_sentiment,
    )

    # --- Valence ---
    vmax_valence = vmax_compartido(valence_few_shot, valence_zero_shot)
    graficar_matriz_confusion(
        valence_few_shot, etiquetas_valence, "Blues", "valence_few_shot.pdf",
        vmax=vmax_valence,
    )
    graficar_matriz_confusion(
        valence_zero_shot, etiquetas_valence, "Reds", "valence_zero_shot.pdf",
        vmax=vmax_valence,
    )

    print("\n¡Listo! Todos los PDFs se guardaron en la carpeta:", OUTPUT_DIR)