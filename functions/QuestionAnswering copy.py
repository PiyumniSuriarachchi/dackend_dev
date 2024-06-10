import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import pdfminer.high_level
import spacy
import requests

# Download the spaCy model if not already downloaded
spacy.cli.download("en_core_web_sm")

# Load the English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize the model and tokenizer once outside the function
model_name = "z-uo/bert-qasper"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

def preprocess_pdf_text(file_path_or_url):
    if file_path_or_url.startswith('http://') or file_path_or_url.startswith('https://'):
        response = requests.get(file_path_or_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch PDF from URL: {file_path_or_url}. Status code: {response.status_code}")
        pdf_text = ""
        for page in pdfminer.high_level.extract_pages(response.content):
            for element in page:
                if isinstance(element, pdfminer.layout.LTTextBoxHorizontal):
                    pdf_text += element.get_text()
    else:
        # Treat as local file path
        with open(file_path_or_url, 'rb') as file:
            pdf_text = ""
            for page in pdfminer.high_level.extract_pages(file):
                for element in page:
                    if isinstance(element, pdfminer.layout.LTTextBoxHorizontal):
                        pdf_text += element.get_text()
    return pdf_text

def get_relevant_context(doc, user_question):
    question_tokens = nlp(user_question.lower())
    relevant_sentences = [sentence.text for sentence in doc.sents if any(token.text.lower() in sentence.text.lower() for token in question_tokens)]
    context_snippet = " ".join(relevant_sentences)
    return context_snippet

def answer_question(question, file_path):
    pdf_text = preprocess_pdf_text(file_path)
    doc = nlp(pdf_text)

    # Get relevant context based on the user's question
    context_snippet = get_relevant_context(doc, question)

    # Use the question-answering model
    answer = qa_pipeline({"context": context_snippet, "question": question})

    return answer["answer"]
