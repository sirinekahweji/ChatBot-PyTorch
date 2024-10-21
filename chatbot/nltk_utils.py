import nltk
nltk.download('punkt_tab')
nltk.data.path.append('C:/Users/Lenovo/AppData/Roaming/nltk_data/nltk/tokenize')
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenizes_sentence,all_words):
    pass

words = ["Organize","organizes","organizing"]
stemmed_words=[stem(w) for w in words]
print(stemmed_words)
#a="How are you baby ?"
#print(a)
#a=tokenize(a)
#print(a)
