import nltk
nltk.download('punkt')

def preprocess(filepath):
    with open(filepath) as file:
        data = file.read()
    
    return nltk.tokenize.sent_tokenize(data)