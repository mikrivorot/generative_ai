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


print(cv.vocabulary_)
# {'go': 23, 'great': 27, 'got': 26, 'wat': 90, 'ok': 59, 'free': 19, 'win': 94, 'text': 79, 'txt': 86, 'say': 70, 'already': 0, 'think': 82, 'life': 39, 'hey': 30, 'week': 92, 'back': 6, 'like': 40, 'still': 75, 'send': 72, 'friend': 20, 'prize': 65, 'claim': 10, 'call': 7, 'mobile': 50, 'co': 11, 'home': 32, 'want': 89, 'today': 84, 'cash': 9, 'day': 15, 'reply': 67, 'www': 96, 'right': 68, 'take': 77, 'time': 83, 'message': 47, 'com': 12, 'oh': 58, 'yes': 99, 'make': 45, 'way': 91, 'dont': 17, 'miss': 49, 'ur': 88, 'going': 24, 'da': 14, 'lor': 42, 'meet': 46, 'really': 66, 'know': 35, 'lol': 41, 'love': 43, 'amp': 2, 'let': 38, 'work': 95, 'yeah': 97, 'tell': 78, 'anything': 3, 'thanks': 80, 'uk': 87, 'please': 63, 'msg': 52, 'see': 71, 'pls': 64, 'need': 54, 'tomorrow': 85, 'hope': 33, 'well': 93, 'lt': 44, 'gt': 28, 'get': 21, 'ask': 4, 'morning': 51, 'happy': 29, 'sorry': 74, 'give': 22, 'new': 55, 'find': 18, 'year': 98, 'later': 37, 'pick': 62, 'good': 25, 'come': 13, 'said': 69, 'hi': 31, 'babe': 5, 'im': 34, 'much': 53, 'stop': 76, 'one': 60, 'night': 56, 'service': 73, 'dear': 16, 'thing': 81, 'last': 36, 'min': 48, 'number': 57, 'also': 1, 'care': 8, 'phone': 61}