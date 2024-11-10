from transformers import pipeline

# Specify the model name and revision
model_name = "facebook/bart-large-mnli"  # Model for zero-shot classification
revision = "main"

# Initialize the pipeline with the 'device' argument set to 0 to use GPU (if available)
# If using CPU, set device=-1 (which is the default)
classifier = pipeline("zero-shot-classification", model=model_name, revision=revision, device=-1)

def categorize_text(text: str, candidate_labels=None, top_k=3):
    # Define default candidate labels if none are provided
    if candidate_labels is None:
        candidate_labels = [
            "Sports", "Politics", "Music", "News", "Technology", "Entertainment",
            "Health", "Education", "Science", "Travel", "Food", "Lifestyle",
            "Gaming", "Business", "Environment", "Casual", "Family", "Humor",
            "Religion", "Hypnosis", "Classes", "Learning", "Animals", "Nature", "Risky"
        ]

    if not text:
        raise ValueError("Input text cannot be empty")

    # Perform the zero-shot classification
    result = classifier(text, candidate_labels, multi_label=False)  # Use multi_label instead of multi_class

    # Return top_k categories
    top_categories = result['labels'][:top_k]
    top_scores = result['scores'][:top_k]

    return {'categories': top_categories, 'scores': top_scores}

# Example Usage
text = "The latest innovations in AI and machine learning are transforming industries."
result = categorize_text(text, top_k=3)
print(result)  # Output: {'categories': ['Technology', 'Science', 'Business'], 'scores': [0.89, 0.08, 0.03]}
