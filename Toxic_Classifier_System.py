import pickle
import streamlit as st
import re
import nltk
import spacy
import string
import pickle
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import words
from sklearn.feature_extraction.text import TfidfVectorizer

# loading the trained model
pickle_in = open('toxicity_classifier_model.pkt', 'rb')
classifier = pickle.load(pickle_in)

pickle_in1 = open('tf_idf.pkt', 'rb')
tfidf = pickle.load(pickle_in1)

# Define the mapping of shortforms to their full words
shortforms_mapping = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'd": "i had",
    "i'll": "i will",
    "i'm": "i am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "i have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": "will",
    "didn't": "did not",
    "tryin'": "trying",
}

def replace_shortforms_in_text(text, shortforms_mapping):
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in shortforms_mapping.keys()) + r')\b')
    replaced_text = pattern.sub(lambda x: shortforms_mapping[x.group()], text)
    return replaced_text

def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
  # remove everything except alphabets
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)
# Load the spaCy language model
nlp = spacy.load("en_core_web_lg")

# Lemmatize the 'comment_text' column using spaCy
def apply_lemmatization(text):
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)
def remove_xspace(text):
    # Remove extra whitespaces using wildcards
    text = re.sub(r'\s+', ' ', text)
    return text
def remove_num(text):
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    return text

# pre-process the text
def preprocess_text_single(text):
    # Convert to lowercase
    text = text.lower()

    # Contraction Expansion
    text = replace_shortforms_in_text(text, shortforms_mapping)

    # Remove digits
    text = re.sub(r'\d+', ' ', text)

    # Clean HTML tags
    text = cleanHtml(text)

    # Clean punctuation
    text = cleanPunc(text)

    # Keep only alphabetic characters
    text = keepAlpha(text)

    # Remove stop words
    text = removeStopWords(text)

    # Apply lemmatization
    text = apply_lemmatization(text)

    # Remove extra whitespaces
    text = remove_xspace(text)

    return text

def toxicity_detec(text):
  preprocessed_test_text = preprocess_text_single(text)
  text_tfidf = tfidf.transform([preprocessed_test_text]).toarray()
  prediction = classifier.predict(text_tfidf).toarray()

  return prediction

def main():
    # front end elements of the web page
    html_temp = """
    <div style ="background-color:#3498db;padding:13px">
      <h1 style ="color:black;text-align:center;">Toxicity Classifier System</h1>
    </div>
    """
    # display the front end aspect
    st.markdown(html_temp , unsafe_allow_html = True)

    text_input=st.text_input("Enter your Text")

    result =""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Analyse"):
        result = toxicity_detec(text_input) 
        if result[0][0] == 1:
            label1="!!!"
        else:
            label1="-"
        if result[0][1] == 1:
            label2="!!!"
        else:
            label2="-"
        if result[0][2] == 1:
            label3="!!!"
        else:
            label3="-"
        if result[0][3] == 1:
            label4="!!!"
        else:
            label4="-"
        if result[0][4] == 1:
            label5="!!!"
        else:
            label5="-"
        if result[0][5] == 1:
            label6="!!!"
        else:
            label6="-"
        if result[0][6] == 1:
            label7=":)"
        else:
            label7="-"
        st.markdown(
        f'<div style="background-color:{"#ffcccb" if result[0][0] == 1 else "inherit"}; padding:10px;">Toxic: {label1}</div>'
        f'<div style="background-color:{"#ffcccb" if result[0][1] == 1 else "inherit"}; padding:10px;">Severe Toxic: {label2}</div>'
        f'<div style="background-color:{"#ffcccb" if result[0][2] == 1 else "inherit"}; padding:10px;">Obscene: {label3}</div>'
        f'<div style="background-color:{"#ffcccb" if result[0][3] == 1 else "inherit"}; padding:10px;">Threat: {label4}</div>'
        f'<div style="background-color:{"#ffcccb" if result[0][4] == 1 else "inherit"}; padding:10px;">Insult: {label5}</div>'
        f'<div style="background-color:{"#ffcccb" if result[0][5] == 1 else "inherit"}; padding:10px;">Identity Hate: {label6}</div>'
        f'<div style="background-color:{"#90ee90" if result[0][6] == 1 else "inherit"}; padding:10px;">Non-Toxic: {label7}</div>',
        unsafe_allow_html=True
        )  

if __name__=='__main__':
  main()