from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from text_preprocessing import load_sms_dataset, prepare_corpus, get_word_frequencies

# Load and prepare data
messages = load_sms_dataset('files/SMSSpamCollection.txt')
corpus = prepare_corpus(messages)

# Create vectors using TF-IDF
tfidf = TfidfVectorizer(max_features=100, ngram_range=(2,2))
X = tfidf.fit_transform(corpus).toarray()

# Print results
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
print(X)
#  [0 0 0 0 0 0 0 0 0 0 0 0.588 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0.578 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

# Get and print word frequencies
word_freq_df = get_word_frequencies(X, tfidf)
print("\nTop 20 most frequent bigrams:")
print(word_freq_df.head(20))