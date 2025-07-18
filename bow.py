# https://github.com/krishnaik06/SpamClassifier/blob/master/smsspamcollection/SMSSpamCollection
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from nltk.stem import WordNetLemmatizer


print("Loading SMS Spam Collection")

try: 
    messages = pd.read_csv('files/SMSSpamCollection.txt', 
                        sep='\t',
                        names=["label", "message"],
                        encoding='utf-8')  
except FileNotFoundError:
    print("Error: SMS Collection file not found")
except Exception as e:
    print(f"Error loading file: {e}")
    

# Step 2. Prepare the data
nltk.download('stopwords')

nltk.download('wordnet')  # Required for lemmatization
nltk.download('omw-1.4')  # Required for wordnet
lemmatizer = WordNetLemmatizer()

# Array of corpuses = list of messages (phrases)
corpus = []
for i in range(0, len(messages)):
                    review = re.sub('[^a-zA-z]', ' ', messages['message'][i])
                    review = review.lower()
                    review = review.split()
                    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
                    review = ' '.join(review)
                    corpus.append(review)
    
# Step 3. Create the bag of words.
# max_features=100 means length of the vocabulary is 100
cv=CountVectorizer(max_features=100,binary=True)
X=cv.fit_transform(corpus).toarray()   

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
print(X)

# [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 ...