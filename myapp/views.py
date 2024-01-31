from django.shortcuts import render, HttpResponse
import os
import sys
import logging
import string
import pandas as pd
import ssl
import certifi
import nltk
# ssl._create_default_https_context = ssl._create_unverified_context

# # Set the SSL CA bundle location
# nltk.data.path.append(certifi.where())
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from spellchecker import SpellChecker
import re
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse, FileResponse
import json
import base64
import fitz  # PyMuPDF
import ocrmypdf
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
import numpy as np

def home(request):
  return HttpResponse("Hello, world. The app is running.")

# Checking if the ID column is unique
def check_unique_ids(df, id_column):
  if df[id_column].nunique() != len(df):
    return 'true'
  else: 
    return 'false'


# Checking if there are atleast 300 rows
def check_minimum_rows(df, min_rows=300):
  if len(df) < min_rows:
    return 'true'
  else: 
    return 'false'


# Checking if there are more than 10% of blank content
def check_blank_content(df, field_to_analyze, max_blank_percentage=10):
  blank_rows = df[field_to_analyze].isnull().sum()
  blank_percentage = (blank_rows / len(df)) * 100

  if blank_percentage > max_blank_percentage:
    return 'true'
  else: 
    return 'false'


# Function to process the file
def preprocess_text(df, field_to_analyze):
  # Initialize a new column with the original text while converting it to lowercase.
  df['processed_text'] = df[field_to_analyze].str.lower()

  # Remove punctuation
  PUNCT_TO_REMOVE = string.punctuation

  def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

  df['processed_text'] = df['processed_text'].apply(lambda text: remove_punctuation(text))

  # Remove stopwords
  STOPWORDS = set(stopwords.words('english'))

  def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

  df['processed_text'] = df['processed_text'].apply(lambda text: remove_stopwords(text))

  # Stemming the words
  stemmer = PorterStemmer()

  def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

  df['processed_text'] = df['processed_text'].apply(lambda text: stem_words(text))

  # Lemmatization of words
  lemmatizer = WordNetLemmatizer()

  def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

  df['processed_text'] = df['processed_text'].apply(lambda text: lemmatize_words(text))

  # Removing emojis
  def remove_emoji(text):
    emoji_pattern = re.compile("["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
      u"\U0001F680-\U0001F6FF"  # transport & map symbols
      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

  df['processed_text'] = df['processed_text'].apply(lambda text: remove_emoji(text))

  # Removing emoticons
  EMOTICONS = {
      r":‑\)": "Happy face or smiley",
      r":\)": "Happy face or smiley",
      r":-\]": "Happy face or smiley",
      r":\]": "Happy face or smiley",
      r":-3": "Happy face smiley",
      r":3": "Happy face smiley",
      r":->": "Happy face smiley",
      r":>": "Happy face smiley",
      r"8-\)": "Happy face smiley",
      r":o\)": "Happy face smiley",
      r":-\}": "Happy face smiley",
      r":\}": "Happy face smiley",
      r":-\)": "Happy face smiley",
      r":c\)": "Happy face smiley",
      r":\^\)": "Happy face smiley",
      r"=\]": "Happy face smiley",
      r"=\)": "Happy face smiley",
      r":‑D": "Laughing, big grin or laugh with glasses",
      r":D": "Laughing, big grin or laugh with glasses",
      r"8‑D": "Laughing, big grin or laugh with glasses",
      r"8D": "Laughing, big grin or laugh with glasses",
      r"X‑D": "Laughing, big grin or laugh with glasses",
      r"XD": "Laughing, big grin or laugh with glasses",
      r"=D": "Laughing, big grin or laugh with glasses",
      r"=3": "Laughing, big grin or laugh with glasses",
      r"B\^D": "Laughing, big grin or laugh with glasses",
      r":-\)\)": "Very happy",
      r":‑\(": "Frown, sad, angry or pouting",
      r":-\(": "Frown, sad, angry or pouting",
      r":\(": "Frown, sad, angry or pouting",
      r":‑c": "Frown, sad, angry or pouting",
      r":c": "Frown, sad, angry or pouting",
      r":‑<": "Frown, sad, angry or pouting",
      r":<": "Frown, sad, angry or pouting",
      r":‑\[": "Frown, sad, angry or pouting",
      r":\[": "Frown, sad, angry or pouting",
      r":-\|\|": "Frown, sad, angry or pouting",
      r">:\[": "Frown, sad, angry or pouting",
      r":\{": "Frown, sad, angry or pouting",
      r":@": "Frown, sad, angry or pouting",
      r">:\(": "Frown, sad, angry or pouting",
      r":'‑\(": "Crying",
      r":'\(": "Crying",
      r":'‑\)": "Tears of happiness",
      r":'\)": "Tears of happiness",
      r"D‑':": "Horror",
      r"D:<": "Disgust",
      r"D:": "Sadness",
      r"D8": "Great dismay",
      r"D;": "Great dismay",
      r"D=": "Great dismay",
      r"DX": "Great dismay",
      r":‑O": "Surprise",
      r":O": "Surprise",
      r":‑o": "Surprise",
      r":o": "Surprise",
      r":-0": "Shock",
      r"8‑0": "Yawn",
      r">:O": "Yawn",
      r":-\*": "Kiss",
      r":\*": "Kiss",
      r":X": "Kiss",
      r";‑\)": "Wink or smirk",
      r";\)": "Wink or smirk",
      r"\*-\)": "Wink or smirk",
      r"\*\)": "Wink or smirk",
      r";‑\]": "Wink or smirk",
      r";\]": "Wink or smirk",
      r";\^\)": "Wink or smirk",
      r":‑,": "Wink or smirk",
      r";D": "Wink or smirk",
      r":‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r":P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r"X‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r"XP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r":‑Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r":Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r":b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r"d:": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r"=p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r">:P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r":‑/": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r":-[.]": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r">:[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r">:/": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r":[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r"=/": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r"=[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r"=L": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r":S": "Skeptical, annoyed, undecided, uneasy or hesitant",
      r":‑\|": "Straight face",
      r":\|": "Straight face",
      r":$": "Embarrassed or blushing",
      r":‑x": "Sealed lips or wearing braces or tongue-tied",
      r":x": "Sealed lips or wearing braces or tongue-tied",
      r":‑#": "Sealed lips or wearing braces or tongue-tied",
      r":#": "Sealed lips or wearing braces or tongue-tied",
      r":‑&": "Sealed lips or wearing braces or tongue-tied",
      r":&": "Sealed lips or wearing braces or tongue-tied",
      r"O:‑\)": "Angel, saint or innocent",
      r"O:\)": "Angel, saint or innocent",
      r"0:‑3": "Angel, saint or innocent",
      r"0:3": "Angel, saint or innocent",
      r"0:‑\)": "Angel, saint or innocent",
      r"0:\)": "Angel, saint or innocent",
      r":‑b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
      r"0;\^\)": "Angel, saint or innocent",
      r">:‑\)": "Evil or devilish",
      r">:\)": "Evil or devilish",
      r"\}:‑\)": "Evil or devilish",
      r"\}:\)": "Evil or devilish",
      r"3:‑\)": "Evil or devilish",
      r"3:\)": "Evil or devilish",
      r">;\)": "Evil or devilish",
      r"\|;‑\)": "Cool",
      r"\|‑O": "Bored",
      r":‑J": "Tongue-in-cheek",
      r"#‑\)": "Party all night",
      r"%‑\)": "Drunk or confused",
      r"%\)": "Drunk or confused",
      r":-###..": "Being sick",
      r":###..": "Being sick",
      r"<:‑\|": "Dump",
      r"\(>_<\)": "Troubled",
      r"\(>_<\)>": "Troubled",
      r"\(';'\)": "Baby",
      r"\(\^\^>``": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
      r"\(\^_\^;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
      r"\(-_-;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
      r"\(~_~;\) \(・\.・;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
      r"\(-_-\)zzz": "Sleeping",
      r"\(\^_-\)": "Wink",
      r"\(\(\+_\+\)\)": "Confused",
      r"\(\+o\+\)": "Confused",
      r"\(o\|o\)": "Ultraman",
      r"\^_\^": "Joyful",
      r"\(\^_\^\)/": "Joyful",
      r"\(\^O\^\)／": "Joyful",
      r"\(\^o\^\)／": "Joyful",
      r"\(__\)": "Kowtow as a sign of respect, or dogeza for apology",
      r"_\(\._\.\)_": "Kowtow as a sign of respect, or dogeza for apology",
      r"<\(_ _\)>": "Kowtow as a sign of respect, or dogeza for apology",
      r"<m\(__\)m>": "Kowtow as a sign of respect, or dogeza for apology",
      r"m\(__\)m": "Kowtow as a sign of respect, or dogeza for apology",
      r"m\(_ _\)m": "Kowtow as a sign of respect, or dogeza for apology",
      r"\('_'\)": "Sad or Crying",
      r"\(/_;\)": "Sad or Crying",
      r"\(T_T\) \(;_;\)": "Sad or Crying",
      r"\(;_;": "Sad of Crying",
      r"\(;_:\)": "Sad or Crying",
      r"\(;O;\)": "Sad or Crying",
      r"\(:_;\)": "Sad or Crying",
      r"\(ToT\)": "Sad or Crying",
      r";_;": "Sad or Crying",
      r";-;": "Sad or Crying",
      r";n;": "Sad or Crying",
      r";;": "Sad or Crying",
      r"Q\.Q": "Sad or Crying",
      r"T\.T": "Sad or Crying",
      r"QQ": "Sad or Crying",
      r"Q_Q": "Sad or Crying",
      r"\(-\.-\)": "Shame",
      r"\(-_-\)": "Shame",
      r"\(一一\)": "Shame",
      r"\(；一_一\)": "Shame",
      r"\(=_=\)": "Tired",
      r"\(=\^\·\^=\)": "cat",
      r"\(=\^\·\·\^=\)": "cat",
      r"=_\^=	": "cat",
      r"\(\.\.\)": "Looking down",
      r"\(\._\.\)": "Looking down",
      r"\^m\^": "Giggling with hand covering mouth",
      r"\(\・\・?": "Confusion",
      r"\(?_?\)": "Confusion",
      r">\^_\^<": "Normal Laugh",
      r"<\^!\^>": "Normal Laugh",
      r"\^/\^": "Normal Laugh",
      r"\（\*\^_\^\*）": "Normal Laugh",
      r"\(\^<\^\) \(\^\.\^\)": "Normal Laugh",
      r"\(^\^\)": "Normal Laugh",
      r"\(\^\.\^\)": "Normal Laugh",
      r"\(\^_\^\.\)": "Normal Laugh",
      r"\(\^_\^\)": "Normal Laugh",
      r"\(\^\^\)": "Normal Laugh",
      r"\(\^J\^\)": "Normal Laugh",
      r"\(\*\^\.\^\*\)": "Normal Laugh",
      r"\(\^—\^\）": "Normal Laugh",
      r"\(#\^\.\^#\)": "Normal Laugh",
      r"\（\^—\^\）": "Waving",
      r"\(;_;\)/~~~": "Waving",
      r"\(\^\.\^\)/~~~": "Waving",
      r"\(-_-\)/~~~ \($\·\·\)/~~~": "Waving",
      r"\(T_T\)/~~~": "Waving",
      r"\(ToT\)/~~~": "Waving",
      r"\(\*\^0\^\*\)": "Excited",
      r"\(\*_\*\)": "Amazed",
      r"\(\*_\*;": "Amazed",
      r"\(\+_\+\) \(@_@\)": "Amazed",
      r"\(\*\^\^\)v": "Laughing,Cheerful",
      r"\(\^_\^\)v": "Laughing,Cheerful",
      r"\(\(d[-_-]b\)\)": "Headphones,Listening to music",
      r'\(-"-\)': "Worried",
      r"\(ーー;\)": "Worried",
      r"\(\^0_0\^\)": "Eyeglasses",
      r"\(\＾ｖ\＾\)": "Happy",
      r"\(\＾ｕ\＾\)": "Happy",
      r"\(\^\)o\(\^\)": "Happy",
      r"\(\^O\^\)": "Happy",
      r"\(\^o\^\)": "Happy",
      r"\)\^o\^\(": "Happy",
      r":O o_O": "Surprised",
      r"o_0": "Surprised",
      r"o\.O": "Surpised",
      r"\(o\.o\)": "Surprised",
      r"oO": "Surprised",
      r"\(\*￣m￣\)": "Dissatisfied",
      r"\(‘A`\)": "Snubbed or Deflated"
  }

  def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

  df['processed_text'] = df['processed_text'].apply(lambda text: remove_emoticons(text))

  # Remove URLs
  def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

  df['processed_text'] = df['processed_text'].apply(lambda text: remove_urls(text))

  # Remove HTML tags
  def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

  df['processed_text'] = df['processed_text'].apply(lambda text: remove_html(text))

  # Spellchecker
  spell = SpellChecker()

  def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
      if word in misspelled_words:
        corrected_word = spell.correction(word)
        if corrected_word is not None:
          corrected_text.append(corrected_word)
        else:
          corrected_text.append(word)
      else:
        corrected_text.append(word)
    return " ".join(corrected_text)

  df['processed_text'] = df['processed_text'].apply(lambda text: correct_spellings(text))

  # Rename the processed text column to 'processed_text'
  # df.rename(columns={field_to_analyze: 'processed_text'}, inplace=True)

  # Uncomment this section if the original inputs needs to be replaced with the processed ones and also remove the line
  # df['processed_text'] = df[field_to_analyze].str.lower()
  # Replace all df['processed_text'] , with df[field_to_analyze].

  return df

# Main function
def process_file_main(input_filepath):
  logging.info(f"Starting process with the input file: {input_filepath}")

  # Ensure the input file exists and is a valid Excel or CSV file
  if not os.path.exists(input_filepath):
    logging.error(f"File {input_filepath} does not exist.")
    sys.exit(1)

  # Validate the file extension
  valid_extensions = ['.csv', '.xls', '.xlsx']
  file_extension = os.path.splitext(input_filepath)[1].lower()
  if file_extension not in valid_extensions:
    logging.error(f"Unsupported file format: {file_extension}. Please provide a CSV, XLS, or XLSX file.")
    sys.exit(1)

  # # Create output directories
  # root_dir = os.path.dirname(input_filepath)
  # output_dir = os.path.join(root_dir, 'processed_output')
  # if not os.path.exists(output_dir):
  #     os.makedirs(output_dir)
  # excel_output_path = os.path.join(root_dir, 'output_data.xlsx')

  # Read the input data based on file extension
  file_extension = os.path.splitext(input_filepath)[1].lower()

  try:
    logging.info(f"Reading the input file: {input_filepath}")
    if file_extension == '.csv':
        df = pd.read_csv(input_filepath)
    else:  # Handle Excel files
        df = pd.read_excel(input_filepath)
    logging.info("Successfully read the input file.")
  except Exception as e:
    logging.error(f"Failed to read the input file: {str(e)}")
    sys.exit(1)

  # Identify the ID and Content columns dynamically
  id_column, field_to_analyze = df.columns[:2]

  # Perform the Text Pre-Processing
  df = preprocess_text(df, field_to_analyze)

  try:
    # Attempt to unlink or delete the file
    os.remove(input_filepath)
    print(f"File {input_filepath} unlinked successfully.")
  except FileNotFoundError:
    print(f"File {input_filepath} not found.")
  except PermissionError:
    print(f"Permission error: Unable to unlink {input_filepath}.")
  except Exception as e:
    print(f"An error occurred: {e}")
  # # Save the processed data to a new Excel file
  # processed_output_path = os.path.join(output_dir, 'Processed.xlsx')
  # df.to_excel('Processed.xlsx', index=False)

  # print("Processing completed. The processed file has been saved at:", processed_output_path)
  return df

# View to process the request file
def process_file(request):
  if request.method == 'POST':

    data_from_frontend = json.loads(request.body)

    # Extracting the 'file' object
    file_data = data_from_frontend['file']
    file_name = data_from_frontend['name']
    file_extention = data_from_frontend['extention']

    # Extracting the 'data' array from the 'file' object
    buffer_data = file_data.get('data', [])

    # Convert the 'data' array to bytes
    bytes_data = bytes(buffer_data)

    # Writing to a binary file using streams
    file_path_binary = f"{file_name}.{file_extention}"

    # Using with statement to automatically close the file
    with open(file_path_binary, 'wb') as file:
      # Write binary data to the file
      data = bytes_data 
      file.write(data)

    out_text = process_file_main(file_path_binary)
    print(out_text, 'output>>>>>>>>>')
    output_filepath = 'Processed.xlsx'
    out_text.to_excel(output_filepath, index=False)
    
    with open(output_filepath, 'rb') as file:
      file_content = file.read()
      # Convert csv_bytes into base64String
      base64_encoded_string = base64.b64encode(file_content).decode('utf-8')
      os.remove(output_filepath)

    
    return JsonResponse({
      'status': 200, 
      'response': base64_encoded_string
    })
  else:
    return JsonResponse({'error': 'Unsupported method'}, status=405)
  
# View to check if the request file is in correct format
def check_file(request):
  if request.method == 'POST':

    data_from_frontend = json.loads(request.body)

    # Extracting the 'file' object
    file_data = data_from_frontend['file']
    file_name = data_from_frontend['name']
    file_extention = data_from_frontend['extention']

    # Extracting the 'data' array from the 'file' object
    buffer_data = file_data.get('data', [])

    # Convert the 'data' array to bytes
    bytes_data = bytes(buffer_data)

    # Writing to a binary file using streams
    file_path_binary = f"{file_name}.{file_extention}"

    # Using with statement to automatically close the file
    with open(file_path_binary, 'wb') as file:
      # Write binary data to the file
      data = bytes_data 
      file.write(data)

    # Read the input data based on file extension
    file_extension = os.path.splitext(file_path_binary)[1].lower()

    try:
      if file_extension == '.csv':
        df = pd.read_csv(file_path_binary)
      else:  # Handle Excel files
        df = pd.read_excel(file_path_binary)
    except Exception as e:
      sys.exit(1)

    # Identify the ID and Content columns dynamically
    id_column, field_to_analyze = df.columns[:2]

    # Check if all values in the 'ID' column are unique
    uniquie_ids_check = check_unique_ids(df, id_column)

    # Check if there are at least 300 rows
    minimum_rows_check = check_minimum_rows(df)

    # Check for blank rows in the 'Content' column
    blank_content_check = check_blank_content(df, field_to_analyze)
    os.remove(file_path_binary)
    
    return JsonResponse({
      'status': 200, 
      'response': {
        'isNotUnique': uniquie_ids_check,
        'hasNotMinimumRows': minimum_rows_check,
        'hasBlankContent': blank_content_check,
      }
    })
  else:
    return JsonResponse({'error': 'Unsupported method'}, status=405)

# Add ocr layer on the pdf
def ocr_pdf(input_path):
  # Perform OCR using ocrmypdf
  ocrmypdf.ocr(input_path, input_path, force_ocr=True)
  return input_path

# Extract text from pdf
def extract_text_from_pdf(pdf_path):
  # Open the PDF file
  with fitz.open(pdf_path) as pdf:
    text = ""
    # Iterate over each page
    for page in pdf:
      # Extract text from the page
      text += page.get_text()
    return text
  
# View to check if the request file is in correct format
def extract_text(request):
  if request.method == 'POST':

    data_from_frontend = json.loads(request.body)

    # Extracting the 'file' object
    file_data = data_from_frontend['file']
    file_name = data_from_frontend['name']
    file_extention = data_from_frontend['extention']

    # Extracting the 'data' array from the 'file' object
    buffer_data = file_data.get('data', [])

    # Convert the 'data' array to bytes
    bytes_data = bytes(buffer_data)

    # Writing to a binary file using streams
    file_path_binary = f"{file_name}.{file_extention}"

    # Using with statement to automatically close the file
    with open(file_path_binary, 'wb') as file:
      # Write binary data to the file
      data = bytes_data 
      file.write(data)
      
    # Perform OCR if needed and get the updated file path
    ocr_pdf_path = ocr_pdf(file_path_binary)
    # os.remove(file_path_binary)

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(ocr_pdf_path)
    print(extracted_text, 'text>>>>>>>>>>>>>>>>>')
    os.remove(ocr_pdf_path)
    
    # with open(ocr_pdf_path, 'rb') as file:
    #   file_content = file.read()
    #   # Convert csv_bytes into base64String
    #   base64_encoded_string = base64.b64encode(file_content).decode('utf-8')
    #   os.remove(ocr_pdf_path)
    
    return JsonResponse({
      'status': 200, 
      'response': extracted_text
    })
  else:
    return JsonResponse({'error': 'Unsupported method'}, status=405)
  
# Extract keywords from content
def extract_keywords(request):
  if request.method == 'POST':

    data_from_frontend = json.loads(request.body)

    # Extracting the 'file' object
    file_data = data_from_frontend['file']
    file_name = data_from_frontend['name']
    file_extention = data_from_frontend['extention']
    key_range = data_from_frontend['keywordRange']

    # Extracting the 'data' array from the 'file' object
    buffer_data = file_data.get('data', [])

    # Convert the 'data' array to bytes
    bytes_data = bytes(buffer_data)

    # Writing to a binary file using streams
    file_path_binary = f"{file_name}.{file_extention}"

    # Using with statement to automatically close the file
    with open(file_path_binary, 'wb') as file:
      # Write binary data to the file
      data = bytes_data 
      file.write(data)
      
    # Read the input data based on file extension
    file_extension = os.path.splitext(file_path_binary)[1].lower()
    if file_extension == '.csv':
      df = pd.read_csv(file_path_binary)
    else:  # Handle Excel files
      df = pd.read_excel(file_path_binary)
      
    def summarize():
      # Create a new column 'Identified Keywords' to store the results
      df['Identified Keywords'] = ''

      # Iterate through each row
      for index, row in df.iterrows():
        # Read content from the row
        review_text = row['text'] or row['Content']
        
        # Create a CountVectorizer to get unique words
        vect = CountVectorizer(stop_words='english')
        vect.fit([review_text])
        unique_words = vect.get_feature_names_out()

        # Create a TfidfVectorizer to calculate TF-IDF scores
        vect_tfidf = TfidfVectorizer(stop_words='english')
        dtm_tfidf = vect_tfidf.fit_transform([review_text])

        # Create a dictionary of words and their TF-IDF scores
        word_scores = {}
        for word in unique_words:
            word_index = np.where(vect_tfidf.get_feature_names_out() == word)[0][0]
            word_scores[word] = dtm_tfidf[0, word_index]

        # Sort the scores and get the top 5 words
        top_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:int(key_range)]

        # Append the results in the 'Identified Keywords' column
        df.at[index, 'Identified Keywords'] = ', '.join([word for word, score in top_scores])

      # Display the modified DataFrame
      return df

    out_text = summarize()
    output_filepath = 'Processed.xlsx'
    out_text.to_excel(output_filepath, index=False)
    
    with open(output_filepath, 'rb') as file:
      file_content = file.read()
      # Convert csv_bytes into base64String
      base64_encoded_string = base64.b64encode(file_content).decode('utf-8')
      os.remove(output_filepath)

   
    return JsonResponse({
      'status': 200, 
      'response': base64_encoded_string
    })
  else:
    return JsonResponse({'error': 'Unsupported method'}, status=405)