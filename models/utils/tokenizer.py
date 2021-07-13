import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['stopwords', 'wordnet', 'punkt'], quiet=True)

def tokenize(text):
    '''
    Processes text to list of tokens
    
    INPUT:
    text
    
    OUTPUT:
    list - tokens generated from the text
    '''
    
    # normalize and tokenize text
    tokens = word_tokenize(text.lower())
    
    # lemmatize and remove stopwords
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    
    return tokens