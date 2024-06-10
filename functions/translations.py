# functions/translations.py
from deep_translator import GoogleTranslator

def split_text(text, chunk_size=500):
    """
    Split a given text into chunks of specified size.
    
    Args:
    text (str): The text to be split into chunks.
    chunk_size (int): The size of each chunk.
    
    Returns:
    list: A list of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def translate_text(data):
    """
    Translate text to the target language using Google Translator.
    
    Args:
    data (dict): A dictionary containing the text to be translated and the target language.
    
    Returns:
    dict: A dictionary containing the translated text.
    """
    try:
        text_to_translate = data.get('text', '')
        target_language = data.get('target_language', 'en')

        # Split text into chunks
        text_chunks = split_text(text_to_translate)

        # Translate each chunk and concatenate results
        translated_text = ''
        for chunk in text_chunks:
            translation = GoogleTranslator(source='auto', target=target_language).translate(chunk)
            translated_text += translation + ' '

        return {'translation': translated_text.strip()}

    except Exception as e:
        return {'error': str(e)}
