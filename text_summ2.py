# Step 1. Importing Libraries
import sys
import math
import bs4 as bs
import urllib.request
import re
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfReader
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

# Execute this line if you are running this code for the first time
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Initializing few variables
lemmatizer = WordNetLemmatizer()
stopWords = set(nltk.corpus.stopwords.words("english"))

# Step 2. Define functions for Reading Input Text

# Function to Read .txt File
def file_text(filepath):
    with open(filepath) as f:
        return f.read().replace("\n", ' ')

# Function to Read PDF File
def pdfReader(pdf_path):
    with open(pdf_path, 'rb') as pdfFileObject:
        pdfReader = PdfReader(pdfFileObject)
        count = len(pdfReader.pages)
        print("\nTotal Pages in pdf = ", count)

        c = input("Do you want to read entire pdf ? [Y]/N : ")
        if c.lower() == 'n':
            start_page = int(input("Enter start page number (Indexing starts from 0): "))
            end_page = int(input(f"Enter end page number (Less than {count}): "))
            if start_page < 0 or start_page >= count or end_page < 0 or end_page >= count:
                print("Invalid page numbers")
                sys.exit()
        else:
            start_page = 0
            end_page = count - 1

        text = ""
        for i in range(start_page, end_page + 1):
            page = pdfReader.pages[i]
            text += page.extract_text()

        return text

# Function to Read Wikipedia Page
def wiki_text(url):
    scrap_data = urllib.request.urlopen(url)
    article = scrap_data.read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = ''.join(p.text for p in paragraphs)
    article_text = re.sub(r'\[[0-9]*\]', '', article_text)
    return article_text

# Step 3. Get Input
input_text_type = int(input("Select input method:\n1. Type or Paste Text\n2. Load from .txt file\n3. Load from .pdf file\n4. From Wikipedia URL\n\n"))

if input_text_type == 1:
    text = input("Enter your text:\n\n")
elif input_text_type == 2:
    text = file_text(input("Enter .txt file path: "))
elif input_text_type == 3:
    text = pdfReader(input("Enter .pdf file path: "))
elif input_text_type == 4:
    text = wiki_text(input("Enter Wikipedia URL: "))
else:
    print("Invalid choice. Exiting.")
    sys.exit()

# Step 4. Summarization Functions

def frequency_matrix(sentences):
    freq_matrix = {}
    for sent in sentences:
        freq_table = {}
        words = [word.lower() for word in word_tokenize(sent) if word.isalnum()]
        for word in words:
            word = lemmatizer.lemmatize(word)
            if word not in stopWords:
                freq_table[word] = freq_table.get(word, 0) + 1
        freq_matrix[sent[:15]] = freq_table
    return freq_matrix

def tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        tf_table = {}
        total_words = sum(freq_table.values())
        for word, count in freq_table.items():
            tf_table[word] = count / total_words
        tf_matrix[sent] = tf_table
    return tf_matrix

def sentences_per_words(freq_matrix):
    word_sentence_count = {}
    for freq_table in freq_matrix.values():
        for word in freq_table:
            word_sentence_count[word] = word_sentence_count.get(word, 0) + 1
    return word_sentence_count

def idf_matrix(freq_matrix, word_sentence_count, total_sentences):
    idf_matrix = {}
    for sent, freq_table in freq_matrix.items():
        idf_table = {}
        for word in freq_table:
            idf_table[word] = math.log10(total_sentences / word_sentence_count[word])
        idf_matrix[sent] = idf_table
    return idf_matrix

def tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf = {}
    for sent in tf_matrix:
        tf_idf_table = {}
        for word in tf_matrix[sent]:
            tf_idf_table[word] = tf_matrix[sent][word] * idf_matrix[sent].get(word, 0)
        tf_idf[sent] = tf_idf_table
    return tf_idf

def score_sentences(tf_idf_matrix):
    sentence_scores = {}
    for sent, word_table in tf_idf_matrix.items():
        total_score = sum(word_table.values())
        if word_table:
            sentence_scores[sent] = total_score / len(word_table)
    return sentence_scores

def average_score(sentence_scores):
    return sum(sentence_scores.values()) / len(sentence_scores)

def create_summary(sentences, sentence_scores, threshold):
    summary = ""
    for sentence in sentences:
        if sentence[:15] in sentence_scores and sentence_scores[sentence[:15]] >= threshold:
            summary += " " + sentence
    return summary

# Step 5. Run Summarizer

# Tokenize into sentences
sentences = sent_tokenize(text)
total_sentences = len(sentences)

# Original word count
original_words = [word for word in word_tokenize(text) if word.isalnum()]
num_words_original = len(original_words)

# Build matrices
freq_matrix = frequency_matrix(sentences)
tf = tf_matrix(freq_matrix)
word_sentence_count = sentences_per_words(freq_matrix)
idf = idf_matrix(freq_matrix, word_sentence_count, total_sentences)
tfidf = tf_idf_matrix(tf, idf)
scores = score_sentences(tfidf)
threshold = average_score(scores)

# Generate summary
summary = create_summary(sentences, scores, 0.8 * threshold)

#Fallback if summary is empty
if not summary.strip():
    print("\nSummary was empty(possibly due to short text). Including full text as summary.")
    summary=" ".join(sentences)

# Output
print("\n\n" + "*"*20 + " Summary " + "*"*20 + "\n")
print(summary)
print("\n" + "-"*50)
print("Total words in original article: ", num_words_original)
print("Total words in summary: ", len(summary.split()))