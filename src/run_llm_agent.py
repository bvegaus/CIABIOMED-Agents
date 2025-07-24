import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

INPUT_FILE = "outputs/explanations/classification_lime_results.json"
OUTPUT_FILE = "outputs/explanations/classification_llm_explanations.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def build_prompt(dataset_name, sample_info):
    sample = sample_info["sample_patient"]
    prediction = sample_info["prediction"]
    lime = sample_info["lime_explanation"]

    lime_lines = []
    for feature, impact in lime:
        sign = "positive" if impact > 0 else "negative"
        lime_lines.append(f"- '{feature}' between {impact:.4f} impact: {sign} ({impact:+.4f})")

    prompt = (
        f"You have a sample from the dataset '{dataset_name}'.\n\n"
        f"Patient features:\n"
        + "\n".join([f"- {k}: {v}" for k, v in sample.items()])
        + f"\n\nThe model predicted class {prediction}.\n"
        f"LIME explanation indicates the following features had the most impact:\n"
        + "\n".join(lime_lines)
        + "\n\nPlease provide a clear and simple explanation for a patient, describing why the model predicted this class."
    )
    return prompt


def generate_explanation(prompt, max_new_tokens=250):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            no_repeat_ngram_size=2,
            num_return_sequences=1
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Free memory
    del inputs
    del outputs
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return text

def calculate_semantic_similarity(explanation, features):
    features_text = " ".join(features)
    emb_exp = embedder.encode(explanation, convert_to_tensor=True)
    emb_feat = embedder.encode(features_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb_exp, emb_feat)
    return similarity.item()

def explain_samples_with_model():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    similarities = []

    for dataset_name, info in data.items():
        print(f"Processing dataset: {dataset_name}")
        for i, sample_info in enumerate(info["samples"]):
            prompt = build_prompt(dataset_name, sample_info)
            explanation = generate_explanation(prompt)
            sample_info["prompt_llm"] = prompt
            sample_info["explanation_llm"] = explanation

            features = [feat for feat, val in sample_info["lime_explanation"]]
            score = calculate_semantic_similarity(explanation, features)
            sample_info["semantic_similarity"] = score

            similarities.append(score)
            print(f"Semantic similarity: {score:.3f}")

            # Save progress after each 
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
                json.dump(data, f_out, indent=2, ensure_ascii=False)

            # Clear memory
            del prompt
            del explanation
            del features
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    print(f"\nAverage global semantic similarity for all samples: {avg_similarity:.3f}")
    print(f"\n✅ Explanations generated and saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    explain_samples_with_model()
