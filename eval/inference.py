import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def visualize_inference(text):
    model_path = "../model" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate probabilities using Softmax
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
    labels = ["Negative", "Positive"]

    # --- Visualization Code ---
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#ff9999', '#66b3ff'] # Red for negative, Blue for positive
    
    bars = ax.barh(labels, probabilities, color=colors)
    ax.set_xlim(0, 1) # Probability is always between 0 and 1
    ax.set_xlabel('Probability Score')
    ax.set_title(f'Sentiment Analysis Results\n"{text[:50]}..."')

    # Add text labels on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2%}', 
                va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    user_input = input("Enter movie review: ")
    visualize_inference(user_input)