from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from text_preprocessing import load_sms_dataset, prepare_corpus, get_word_frequencies

# Load and prepare data
messages = load_sms_dataset('files/SMSSpamCollection.txt')
corpus = prepare_corpus(messages)

# Create the bag of words
cv = CountVectorizer(max_features=100, binary=True)
X = cv.fit_transform(corpus).toarray()

# Print results
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
print(X)

# Get and print word frequencies
word_freq_df = get_word_frequencies(X, cv)
print("\nTop 20 most frequent words:")
print(word_freq_df.head(20))