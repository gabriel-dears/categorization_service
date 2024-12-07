from transformers import pipeline

# Specify the model name and revision
model_name = "facebook/bart-large-mnli"  # Model for zero-shot classification
revision = "main"

# Initialize the pipeline with the 'device' argument set to 0 to use GPU (if available)
# If using CPU, set device=-1 (which is the default)
classifier = pipeline("zero-shot-classification", model=model_name, revision=revision, device=-1)


def categorize_text_with_tags_and_category(text: str, tags=None, category=None, candidate_labels=None, top_k=5):
    # Define default candidate labels in Portuguese if none are provided
    if candidate_labels is None:
        candidate_labels = [
            "Esportes", "Política", "Música", "Notícias", "Tecnologia", "Entretenimento",
            "Saúde", "Educação", "Ciência", "Viagem", "Comida", "Estilo de Vida",
            "Jogos", "Negócios", "Meio Ambiente", "Casual", "Família", "Humor",
            "Religião", "Hipnose", "Aulas", "Aprendizagem", "Animais", "Natureza", "Arriscado"
        ]

    # Add the video category to the candidate labels if provided
    if category and category not in candidate_labels:
        candidate_labels.append(category)

    # If tags are provided, include them in the candidate labels
    if tags:
        for tag in tags:
            if tag not in candidate_labels:
                candidate_labels.append(tag)

    if not text:
        raise ValueError("Input text cannot be empty")

    # Perform the zero-shot classification
    result = classifier(text, candidate_labels, multi_label=False)

    # Combine categories and scores into a list of dictionaries
    categorized_results = [
        {"category": label, "score": score}
        for label, score in zip(result['labels'][:top_k], result['scores'][:top_k])
    ]

    return categorized_results



# Example Usage
text = "As últimas inovações em IA e aprendizado de máquina estão transformando as indústrias."
tags = ["IA", "Tecnologia", "Inovação"]
category = "Tecnologia"  # e.g., category retrieved from YouTube API
result = categorize_text_with_tags_and_category(text, tags, category, top_k=5)
print(result)  # Output: {'categories': ['Tecnologia', 'Ciência', 'Negócios'], 'scores': [0.89, 0.08, 0.03]}
