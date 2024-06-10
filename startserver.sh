
# Install required Python packages
pip install flask flask_cors deep_translator transformers scikit-learn sentencepiece pdfplumber nltk datasets pdfminer.six spacy Pillow || exit
python3 -m spacy download en_core_web_sm

# Run the main Python script
python3 main.py || exit
