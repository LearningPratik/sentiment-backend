import re
import string
import nltk

# from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk
# nltk.data.path.append("./nltk_data")
# from nltk.corpus import stopwords

# STOPWORDS = stopwords.words("english")

stemmer = PorterStemmer()

def clean_text(text):
    try:

        # lower the word
        text = text.lower()

        # using regex for removing unnecessary parts
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\w*\d\w*", "", text)
        
        # tokenizing / split the words 
        text = nltk.word_tokenize(text)
        
        # remove stopwords
        # text = [t for t in text if t not in STOPWORDS]
        
        # apply stemming
        text = [stemmer.stem(t) for t in text]

        return " ".join(text)
    
    except Exception:
        return ""
