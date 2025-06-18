import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

def cargar_datos(nombre_dataset):
    if nombre_dataset == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
    elif nombre_dataset == "diabetes":
        data = load_diabetes(as_frame=True)
        X, y = data.data, (data.target > data.target.median()).astype(int)  # Clasificación binaria
    else:
        raise ValueError(f"Dataset {nombre_dataset} no soportado.")
    return X, y
