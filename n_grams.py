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
    
# Step 3. Create vectors using n-grams: bi-grams + trigrams
cv=CountVectorizer(max_features=100,binary=True,ngram_range=(2,3))
X=cv.fit_transform(corpus).toarray()

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
print(X)
# [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 0 0 ...

print(cv.vocabulary_)
# {'free entry': 31, 'claim call': 18, 'call claim': 3, 'free call': 30, 'chance win': 17, 'txt word': 90, 'let know': 53, 'please call': 67, 'lt gt': 59, 'want go': 97, 'like lt': 54, 'like lt gt': 55, 'sorry call': 82, 'call later': 11, 'sorry call later': 83, 'ur awarded': 91, 'call customer': 4, 'customer service': 23, 'cash prize': 16, 'call customer service': 5, 'po box': 70, 'trying contact': 88, 'draw show': 27, 'show prize': 80, 'prize guaranteed': 74, 'guaranteed call': 40, 'valid hr': 95, 'draw show prize': 28, 'show prize guaranteed': 81, 'prize guaranteed call': 75, 'selected receive': 77, 'private account': 71, 'account statement': 0, 'call identifier': 6, 'identifier code': 47, 'code expires': 22, 'private account statement': 72, 'call identifier code': 7, 'identifier code expires': 48, 'urgent mobile': 94, 'call landline': 10, 'wat time': 98, 'ur mob': 93, 'gud ni': 42, 'new year': 63, 'send stop': 79, 'co uk': 21, 'nice day': 64, 'lt decimal': 57, 'decimal gt': 25, 'lt decimal gt': 58, 'good morning': 34, 'ur friend': 92, 'good night': 35, 'reply call': 76, 'last night': 52, 'camera phone': 15, 'pick phone': 66, 'pls send': 68, 'send message': 78, 'pls send message': 69, 'great day': 36, 'suite land': 84, 'land row': 51, 'suite land row': 85, 'take care': 86, 'call mobileupd': 12, 'call optout': 13, 'gt min': 39, 'lt gt min': 60, 'txt stop': 89, 'dating service': 24, 'call land': 8, 'land line': 49, 'line claim': 56, 'claim valid': 19, 'guaranteed call land': 41, 'call land line': 9, 'land line claim': 50, 'claim valid hr': 20, 'gt lt': 37, 'gt lt gt': 38, 'hope good': 46, 'free text': 32, 'holiday cash': 45, 'prize claim': 73, 'nd attempt': 62, 'attempt contact': 1, 'ok lor': 65, 'want come': 96, 'every week': 29, 'happy new': 43, 'happy new year': 44, 'national rate': 61, 'week txt': 99, 'tell ur': 87, 'gift voucher': 33, 'await collection': 2, 'dont know': 26, 'call per': 14}

word_freq = np.sum(X, axis=0)

freq_dict = {}
for word, idx in cv.vocabulary_.items():
                    freq_dict[word] = word_freq[idx]

# Convert to DataFrame and sort by frequency
word_freq_df = pd.DataFrame.from_dict(freq_dict, orient='index', columns=['frequency'])
word_freq_df = word_freq_df.sort_values(by='frequency', ascending=False)

# Display top 20 most frequent n-grams
print("\nTop 20 most frequent n-grams:")
print(word_freq_df.head(20))

# Top 20 most frequent n-grams:
#                        frequency
# lt gt                        214
# please call                   55
# call later                    52
# co uk                         47
# let know                      40
# take care                     40
# sorry call                    38
# sorry call later              38
# po box                        33
# good morning                  33
# new year                      31
# customer service              29
# call landline                 25
# guaranteed call               23
# ok lor                        22
# prize guaranteed              22
# pls send                      22
# prize guaranteed call         21
# gt min                        21
# decimal gt                    21