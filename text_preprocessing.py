import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd

def load_sms_dataset(file_path):
    """Load SMS dataset from file."""
    try: 
        messages = pd.read_csv(file_path, 
                            sep='\t',
                            names=["label", "message"],
                            encoding='utf-8')  
        return messages
    except FileNotFoundError:
        print("Error: SMS Collection file not found")
    except Exception as e:
        print(f"Error loading file: {e}")
    return None

def prepare_corpus(messages):
    """Prepare corpus from messages."""
    # Download required NLTK data
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    lemmatizer = WordNetLemmatizer()
    corpus = []
    
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-z]', ' ', messages['message'][i])
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    
    return corpus

def get_word_frequencies(X, vectorizer):
    """Get word frequencies sorted in descending order."""
    word_freq = np.sum(X, axis=0)
    freq_dict = {word: word_freq[idx] for word, idx in vectorizer.vocabulary_.items()}
    
    word_freq_df = pd.DataFrame.from_dict(freq_dict, orient='index', columns=['frequency'])
    return word_freq_df.sort_values(by='frequency', ascending=False)