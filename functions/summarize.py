from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch

def summarize_text(text):
    try:
        # Check if a GPU is available and use it, otherwise use the CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        # Load the t5-small model and tokenizer on the specified device
        model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        # Tokenize and summarize the input on the specified device
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
        summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, temperature=0.75)

        # Decode the summarized output
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {'summary': summary}

    except Exception as e:
        return {'error': str(e)}