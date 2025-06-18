import lime
import lime.lime_tabular
import numpy as np

def explicar_muestra(modelo, X_train, X_sample):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=["Clase 0", "Clase 1"],
        mode="classification"
    )
    exp = explainer.explain_instance(
        data_row=X_sample.values,
        predict_fn=modelo.predict_proba,
        num_features=5
    )
    return exp.as_list()
