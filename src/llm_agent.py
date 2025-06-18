def generar_prompt_explicacion(dataset_name, datos):
    paciente = datos["sample_patient"]
    prediccion = datos["prediction"]
    explicacion = datos["lime_explanation"]

    prompt = f"""
Eres un agente explicativo entrenado para ayudar a profesionales de la salud a entender decisiones tomadas por un modelo de inteligencia artificial.

El modelo ha sido entrenado con datos del conjunto **{dataset_name}** y ha realizado una predicción para un paciente concreto.

---

🔍 **Predicción del modelo**: Clase {prediccion}

📋 **Datos del paciente:**
"""
    for k, v in paciente.items():
        prompt += f"- {k}: {v}\n"

    prompt += "\n📊 **Factores más relevantes según el modelo (LIME):**\n"
    for feature, weight in explicacion:
        direction = "aumenta la probabilidad" if weight > 0 else "disminuye la probabilidad"
        prompt += f"- {feature} ({direction}, peso: {round(abs(weight), 2)})\n"

    prompt += """
---

🧠 Redacta una explicación comprensible para un profesional sanitario sobre por qué el modelo ha tomado esta decisión, utilizando lenguaje claro y evitando jerga matemática. Explica la influencia de los factores más importantes.
"""
    return prompt
