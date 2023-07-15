from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def remove_punctuations(text):
    text_no_punct = text.translate(str.maketrans("", "", string.punctuation))
    return text_no_punct


def n_grams(text, n):
    tokens = word_tokenize(text)
    if n == 1:
        return tokens
    else:
        n_grams = list(ngrams(tokens, n))
        return n_grams


def stem(text):
    st = PorterStemmer()
    stems = []

    for word in text:
        if isinstance(word, str):
            stems.append(st.stem(word))
        else:
            stems.append([st.stem(i) for i in word])
    return stems


def lem(text):
    l = WordNetLemmatizer()
    lems = []
    for word in text:
        
        if isinstance(word, str):
            lems.append(l.lemmatize(word))
        else:
            
            lems.append([l.lemmatize(i) for i in word])
    return lems


def remove_stopwords(text, sw=[]):
    mat = []
    stop_words = stopwords.words('english')
    stop_words += sw
    

    for word in text:
        if isinstance(word, str):
            if word.lower() not in stop_words:
                mat.append(word)
        else:
            x = []
            for i in word:
                if i.lower() not in stop_words:
                    x.append(i)
            mat.append(x)
    return mat
