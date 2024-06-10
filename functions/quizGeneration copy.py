from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
import random
import re
import torch
import nltk
from nltk.corpus import stopwords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the first model for question and answer generation
qa_tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")

qa_model.to(device)

nltk.download('stopwords')

def generate_unique_distractors(question, answer, context, num_distractors=2):
    # Tokenize the context into words and remove stopwords
    words = set(context.split()) - set(stopwords.words("english"))

    # Convert the question and answer to lowercase
    question_lower = question.lower()
    answer_lower = answer.lower()

    # Ensure question and answer are not in the list of words
    words.discard(question_lower)
    words.discard(answer_lower)

    # Select a random subset of words as unique distractors
    unique_distractors = random.sample(words, min(num_distractors, len(words)))

    # Combine question, answer, and distractors
    return [question, answer] + unique_distractors




def extract_paragraphs(context, min_length=10):
    # Split the text into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\.\s|\?\s|\!\s', context) if len(p.split()) > min_length]

    return paragraphs

def generate_question_answer(context, max_retries=1, min_length=5, max_length=100):
    # Get the device of the QA model
    device = qa_model.device

    for _ in range(max_retries):
        try:
            # Tokenize the context and generate a question-answer pair using the QA model
            inputs = qa_tokenizer(context, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = qa_model.generate(**inputs, max_length=100)
            question_answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=False).replace(qa_tokenizer.pad_token, "").replace(qa_tokenizer.eos_token, "")

            # Check if the separator is present in the generated question-answer pair
            if qa_tokenizer.sep_token in question_answer:
                question, answer = question_answer.split(qa_tokenizer.sep_token, 1)

                # Check if the lengths of the generated question and answer meet the criteria
                if min_length <= len(question) <= max_length and min_length <= len(answer) <= max_length:
                    return question, answer
        except ValueError:
            pass

    # If retries fail, return error messages
    return "Error: Unable to generate question", "Error: Unable to generate answer"

def generate_quiz(context, max_retries=3):
    for _ in range(max_retries):
        # Extract paragraphs from the context
        paragraphs = extract_paragraphs(context)

        # Filter paragraphs based on criteria
        filtered_paragraphs = [p for p in paragraphs if "desired_keyword" in p.lower() and "<unk>" not in p]
        selected_paragraphs = filtered_paragraphs if filtered_paragraphs else paragraphs

        if selected_paragraphs:
            # Randomly select a paragraph
            selected_paragraph = random.choice(selected_paragraphs)

            # Generate question and answer
            question, correct_answer = generate_question_answer(selected_paragraph)

            # Check if the generated question and answer are valid
            if question != "Error: Unable to generate question" and correct_answer != "Error: Unable to generate answer":
                # Generate distractors
                distractors = generate_unique_distractors(question, correct_answer, context)

                # Shuffle options
                random.shuffle(distractors)

                # Check if the question is present in the options and remove it if necessary
                if question in distractors:
                    distractors.remove(question)

                # Ensure the correct answer is included in the options
                if correct_answer not in distractors:
                    distractors.append(correct_answer)

                return question, distractors, correct_answer

    # If all retries fail, return default values or placeholders
    return "Error: Unable to generate question", [], "Error: Unable to generate answer"



def extract_text_without_header_footer(page, top_percentage=10, bottom_percentage=10):
    # Calculate the crop values based on percentages
    top_crop = int(page.height * (top_percentage / 100))
    bottom_crop = int(page.height * (bottom_percentage / 100))

    # Crop the page to exclude a region at the top (header) and bottom (footer)
    cropped_page = page.crop((0, top_crop, page.width, page.height - bottom_crop))

    # Extract text from the cropped page
    return cropped_page.extract_text()

def process_pdf(file_path, max_retries=3):
    for _ in range(max_retries):
        # Read the PDF
        with pdfplumber.open(file_path) as pdf:
            # Exclude the first two pages
            excluded_pages = {0, 1}
            candidate_pages = [i for i in range(len(pdf.pages)) if i not in excluded_pages]

            if not candidate_pages:
                print("No eligible pages found.")
                return "Error: Unable to generate question", "Error: Unable to generate options", "Error: Unable to generate answer"

            # Select a random page from the eligible pages
            random_page_index = random.choice(candidate_pages)
            random_page = pdf.pages[random_page_index]

            # Extract text from the selected page excluding header and footer
            page_text = extract_text_without_header_footer(random_page)

            # Generate quiz based on the page text
            result = generate_quiz(page_text)

            # Check if question and answer are successfully generated
            if result[0] != "Error: Unable to generate question" and result[2] != "Error: Unable to generate answer":
                return result

    # If all retries fail, return default values or placeholders
    print("Max retries reached. Unable to generate question and answer.")
    return "Error: Unable to generate question", "Error: Unable to generate options", "Error: Unable to generate answer"


# Provide the path to your PDF file
# pdf_path = None

# Generate quiz
# question, options, correct_answer = process_pdf(pdf_path)

# Print quiz details
# print("\nQuestion:", question)
# print("Options:", options)
# print("Correct Answer:", correct_answer)


