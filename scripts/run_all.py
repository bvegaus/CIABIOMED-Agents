import json
import os
import pandas as pd
from src.data_preprocessing import cargar_datos
from src.train_models import entrenar_modelo
from src.explainability import explicar_muestra

DATASETS = ["breast_cancer", "diabetes"]
OUTPUT_DIR = "outputs/explanations"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    resultados = {}

    for dataset in DATASETS:
        print(f"Cargando dataset {dataset}...")
        X, y = cargar_datos(dataset)

        print("Entrenando modelo...")
        modelo, X_test, y_test, acc = entrenar_modelo(X, y)

        resultados[dataset] = {
            "accuracy": acc,
            "samples": []
        }

        print(f"Generando explicaciones para todas las muestras de test en {dataset}...")

        for idx in range(len(X_test)):
            sample = X_test.iloc[idx]
            pred = int(modelo.predict([sample])[0])
            expl = explicar_muestra(modelo, X, sample)
            resultados[dataset]["samples"].append({
                "index": int(X_test.index[idx]),
                "sample_patient": sample.to_dict(),
                "prediction": pred,
                "lime_explanation": expl
            })

        print(f"{dataset} listo, accuracy: {acc:.2f}, explicaciones generadas: {len(X_test)}")

    with open(os.path.join(OUTPUT_DIR, "classification_lime_results.json"), "w") as f:
        json.dump(resultados, f, indent=4)

if __name__ == "__main__":
    main()
