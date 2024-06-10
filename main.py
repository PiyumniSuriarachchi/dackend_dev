# Import necessary modules
import os
import csv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from functions.translations import translate_text
from functions.summarize import summarize_text
from functions.quizGeneration import process_pdf
from functions.QuestionAnswering import answer_question

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing


# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
USER_CREDENTIALS_FILE = 'user_credentials.csv'

# Global variable to store the currently viewed PDF
current_viewed_pdf = None


# Endpoint for user sign up
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Check if username already exists
    with open(USER_CREDENTIALS_FILE, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username:
                return jsonify({'error': 'Username already exists'}), 400

    # Append new user to the file
    with open(USER_CREDENTIALS_FILE, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([username, password])

    return jsonify({'message': 'User registered successfully'})

# Endpoint for user login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Check if username and password match any entry in the file
    with open(USER_CREDENTIALS_FILE, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == username and row[1] == password:
                return jsonify({'message': 'Login successful'})
    
    return jsonify({'error': 'Invalid username or password'}), 401


# Endpoint for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for file download
@app.route('/uploads/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

# Endpoint to get the list of uploaded files
@app.route('/getUploadedFiles', methods=['GET'])
def get_uploaded_files():
    try:
        uploaded_files = [{'name': file, 'url': f'/uploads/{file}'} for file in os.listdir(app.config['UPLOAD_FOLDER'])]
        return jsonify(uploaded_files)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for text translation
@app.route('/translate', methods=['POST'])
def translate_handler():
    try:
        data = request.get_json()
        translation_result = translate_text(data)
        return jsonify(translation_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for text summarization
@app.route('/summarize', methods=['POST'])
def summarize_handler():
    try:
        data = request.get_json()
        summary_result = summarize_text(data['text'])
        return jsonify(summary_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint to set the currently viewed PDF
@app.route('/setCurrentlyViewedPDF', methods=['POST'])
def set_currently_viewed_pdf():
    try:
        data = request.get_json()
        global current_viewed_pdf
        # Extract file path from request data
        file_path = data.get('filepath')
        # Extract file name from file path
        file_name = os.path.basename(file_path)
        # Set current_viewed_pdf to the full path of the uploaded PDF
        current_viewed_pdf = "uploads/" + file_name
        return jsonify({'message': 'Currently viewed PDF set successfully', 'filename': current_viewed_pdf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for generating quiz questions
@app.route('/generateQuiz', methods=['GET'])
def generate_quiz():
    global current_viewed_pdf
    # Process the currently viewed PDF to generate quiz questions
    question, options, correct_answer = process_pdf(current_viewed_pdf)
    quiz_data = [question, options, correct_answer]
    return jsonify(quiz_data)

# Endpoint for generating answers to quiz questions
@app.route('/generateAnswer', methods=['POST'])
def generate_answer():
    try:
        global current_viewed_pdf
        data = request.get_json()
        original_text = data['question']
        input_language = data['input_lan']
        # Translate the original question to English
        translation_result = translate_text({'text': original_text, 'target_language': 'en'})
        
        if current_viewed_pdf is None:
            return jsonify({'error': 'No PDF file set for processing'}), 400
        
        question = translation_result['translation']
        # Generate an answer to the translated question using PDF contents
        answer = answer_question(question, current_viewed_pdf)
        return jsonify(answer)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask application using the built-in server
    app.run(host="0.0.0.0", port=8483, debug=True)



# pip install flask flask_cors deep_translator transformers sentencepiece pdfplumber nltk datasets pdfminer.six spacy Pillow
