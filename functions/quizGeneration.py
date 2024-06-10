from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer, BertModel
import pdfplumber
import random
import re
import torch
import nltk
from nltk.corpus import stopwords
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset

# Load spaCy model for tokenization and part-of-speech tagging
nlp = spacy.load("en_core_web_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the QA model for question and answer generation
qa_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
qa_model.to(device)

# Load BERT model and tokenizer for generating distractors
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)

nltk.download('stopwords')

def get_bert_embedding(text, batch_size=8):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [tensor.to(device) for tensor in batch]
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def generate_semantic_distractors(answer, context, num_distractors=2):
    # Tokenize the context using spaCy
    doc = nlp(context)
    
    # Extract unique words from the context
    words = list(set([token.text.lower() for token in doc if token.is_alpha and token.text.lower() != answer.lower()]))
    
    if not words:
        return []
    
    # Get the BERT embedding for the answer
    answer_embedding = get_bert_embedding([answer])[0].reshape(1, -1)
    
    # Get embeddings for words in batches
    word_embeddings = get_bert_embedding(words)
    
    # Calculate similarities
    similarities = cosine_similarity(answer_embedding, word_embeddings).flatten()
    
    # Sort words by similarity and select top distractors
    sorted_indices = similarities.argsort()[::-1]
    distractors = [words[i] for i in sorted_indices[:num_distractors]]
    
    return list(set(distractors))  # Ensure distractors are unique

def extract_paragraphs(context, min_length=10):
    # Split the text into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\.\s|\?\s|\!\s', context) if len(p.split()) > min_length]
    return paragraphs

def generate_question_answer(context, max_retries=1, min_length=5, max_length=100):
    device = qa_model.device

    for _ in range(max_retries):
        try:
            inputs = qa_tokenizer(context, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = qa_model.generate(**inputs, max_length=100)
            question_answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=False).replace(qa_tokenizer.pad_token, "").replace(qa_tokenizer.eos_token, "")

            if qa_tokenizer.sep_token in question_answer:
                question, answer = question_answer.split(qa_tokenizer.sep_token, 1)
                if min_length <= len(question) <= max_length and min_length <= len(answer) <= max_length:
                    return question.strip(), answer.strip()
        except ValueError:
            pass

    return "Error: Unable to generate question", "Error: Unable to generate answer"

def generate_quiz(context, max_retries=3):
    for _ in range(max_retries):
        paragraphs = extract_paragraphs(context)

        filtered_paragraphs = [p for p in paragraphs if "desired_keyword" in p.lower() and "<unk>" not in p]
        selected_paragraphs = filtered_paragraphs if filtered_paragraphs else paragraphs

        if selected_paragraphs:
            selected_paragraph = random.choice(selected_paragraphs)

            question, correct_answer = generate_question_answer(selected_paragraph)

            if question != "Error: Unable to generate question" and correct_answer != "Error: Unable to generate answer":
                distractors = generate_semantic_distractors(correct_answer, context)

                # Ensure the correct answer is included and unique
                distractors = list(set(distractors + [correct_answer.lower()]))
                random.shuffle(distractors)

                return question, distractors, correct_answer

    return "Error: Unable to generate question", [], "Error: Unable to generate answer"

def extract_text_without_header_footer(page, top_percentage=10, bottom_percentage=10):
    top_crop = int(page.height * (top_percentage / 100))
    bottom_crop = int(page.height * (bottom_percentage / 100))

    cropped_page = page.crop((0, top_crop, page.width, page.height - bottom_crop))
    return cropped_page.extract_text()

def process_pdf(file_path, max_retries=3):
    for _ in range(max_retries):
        with pdfplumber.open(file_path) as pdf:
            excluded_pages = {0, 1}
            candidate_pages = [i for i in range(len(pdf.pages)) if i not in excluded_pages]

            if not candidate_pages:
                print("No eligible pages found.")
                return "Error: Unable to generate question", "Error: Unable to generate options", "Error: Unable to generate answer"

            random_page_index = random.choice(candidate_pages)
            random_page = pdf.pages[random_page_index]

            page_text = extract_text_without_header_footer(random_page)

            result = generate_quiz(page_text)

            if result[0] != "Error: Unable to generate question" and result[2] != "Error: Unable to generate answer":
                return result

    print("Max retries reached. Unable to generate question and answer.")
    return "Error: Unable to generate question", "Error: Unable to generate options", "Error: Unable to generate answer"
