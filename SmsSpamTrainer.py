#Imports Here
import argparse
import numpy as np
import pandas as pd
import nltk
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#Command Line Arguments
parser = argparse.ArgumentParser(description='SMS Spam Filter Trainer')
parser.add_argument('-d', '--data_file', metavar='', type=str, help='Path to data directory (labelled & tab-separated)')
#Classifier
parser.add_argument('-r', '--random_forest', action='store_false', help='Use RandomForest Classification')
parser.add_argument('-g', '--gradient_boosting', action='store_true', help='Use Gradient Boosting Classification')
#Classifier Parameters
parser.add_argument('-n', '--n_estimators', metavar='', type=int, help='Define number of estimators')
parser.add_argument('-m', '--max_depth', metavar='', type=int, help='Define max-depth')
#Vectorizer
parser.add_argument('-t', '--tfidf_vectorizer', action='store_false', help='Use TfIdf vectorizer')
parser.add_argument('-c', '--count_vectorizer', action='store_true', help='Use Count vectorizer')
#Preprocessing
parser.add_argument('-s', '--stem_data', action='store_false', help='Use stemming for pre-processing')
parser.add_argument('-l', '--lemmatize_data', action='store_true', help='Use lemmetizing for pre-processing')
args = parser.parse_args()

#FUNCTIONS
def pre_process_stem(sms):
    remove_punct = "".join([word.lower() for word in sms if word not in punctuation])
    tokenize = nltk.tokenize.word_tokenize(remove_punct)
    stemmer = nltk.PorterStemmer()
    stem = [stemmer.stem(word) for word in tokenize if word not in stopwords]
    return stem

def pre_process_lemmatize(sms):
    text = "".join([(word).lower() for word in sms if word not in string.punctuation])
    tokens = re.split('\W+', text)
    lemmatizer = nltk.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return text

def find_features(sms):
    length = sum([1 for letter in sms if letter is not " "])
    capital = sum([1 for letter in sms if letter.isupper()])
    capital = round(capital / length * 100, 2)
    punct = sum([1 for letter in sms if letter in punctuation])
    punct = round(punct / length * 100, 2)
    return length, punct, capital

def drop_null(df):
    null_idx, _ = np.where(pd.isnull(df))
    idx = []
    for i in np.unique(null_idx):
        idx.append(i.item())
    idx.reverse()
    for i in idx:
        df = df.drop(index=i, axis=0)
    df.reset_index(level=[0], drop=True, inplace=True)
    print(' * Removed {} rows with Null values'.format(len(idx)))
    return df

def drop_empty(df):
    empty_idx, _ = np.where(df.applymap(lambda x: x == ''))
    idx = []
    for i in np.unique(empty_idx):
        idx.append(i.item())
    idx.reverse()
    for i in idx:
        df = df.drop(index=i, axis=0)
    df.reset_index(inplace=True, drop=True)
    print(' * Removed {} rows with empty values'.format(len(idx)))
    return df

def vectorize():
    vect_fit = vectorizer.fit(x_train['sms_content'])
    train_transform = vect_fit.transform(x_train['sms_content'])
    test_transform = vect_fit.transform(x_test['sms_content'])

    train_data = pd.concat([x_train[['length', 'punct%', 'capital%']].reset_index(drop=True),
                   pd.DataFrame(train_transform.toarray())], axis=1)
    test_data = pd.concat([x_test[['length', 'punct%', 'capital%']].reset_index(drop=True),
                   pd.DataFrame(test_transform.toarray())], axis=1)
    return train_data, test_data

def classify(train, test, labels):
    start_time = time.time()
    classifier = load_classifier
    model = classifier.fit(train, labels)
    y_pred = model.predict(test)
    precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')
    print('   Classifier: {}\n   Vectorizer: {}\n   Pre-Processing: {}\n   Estimators: {}\n   Depth: {}\n   Model took {} seconds to train'.format(
        str(type(classifier))[32:-2] , str(vectorizer.__class__)[40:-2], str(pre_process.__name__)[12:], n_estimators, max_depth, round(time.time() - start_time, 2) ))
    print('------------------------------------------------------------------------------\n   RESULTS\n------------------------------------------------------------------------------')
    print('   Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))

def dataset_header():
    print('------------------------------------------------------------------------------\n   DATASET\n------------------------------------------------------------------------------')
    print('1. Dataset {} is sucessfully loaded\n   Number of columns: {} / Number of Rows: {}'.format(data_file, sms_dataset.shape[1], sms_dataset.shape[0]))
def training_header():
    print('------------------------------------------------------------------------------\n   TRAINING SPECS\n------------------------------------------------------------------------------')

#Pre-Processing Parameters
punctuation = string.punctuation
stopwords = nltk.corpus.stopwords.words('english')

#Optional String/Int Arguments
data_file = (args.data_file or 'SMSSpamCollection.txt') #concatinate to string by: test_dir = os.path.join(args.data_dir, "test")
n_estimators = (args.n_estimators or 150)
max_depth = (args.max_depth or None)

#Optional Boolean Arguments
pre_process = None
if args.lemmatize_data == True:
    pre_process = pre_process_lemmatize
else:
    pre_process = pre_process_stem

vectorizer = None
if args.count_vectorizer == True:
    vectorizer = CountVectorizer(analyzer=pre_process)
else:
    vectorizer = TfidfVectorizer(analyzer=pre_process)

load_classifier = None
if args.gradient_boosting == True:
    load_classifier = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
else:
    load_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)

#Main function
if __name__ == '__main__':
    #Loading dataset
    sms_dataset = pd.read_csv(data_file, sep = '\t', header=None)
    sms_dataset.columns = ['label', 'sms_content']
    dataset_header()

    #Adding more features to dataset - length of sms, % of punctuation and & of capital letters
    features = sms_dataset['sms_content'].apply(lambda x: find_features(x))
    sms_dataset['length'], sms_dataset['punct%'], sms_dataset['capital%'] = zip(*features)

    #Creating a dataframe
    sms_df = pd.DataFrame(sms_dataset)

    #Checking for missing values
    if sms_df.isnull().values.any() == True:
        sms_df = drop_null(sms_df)
    if sms_df.applymap(lambda x: x == '').sum().any() == True:
        sms_df = drop_empty(sms_df)

    #Splitting Dataset into Training/Test Samples
    x_train, x_test, y_train, y_test = train_test_split(sms_df[['sms_content', 'length', 'punct%', 'capital%']], sms_df['label'], test_size=0.25)

    #Vectorizing data
    train_data, test_data = vectorize()

    #Training Model
    training_header()
    model = classify(train_data, test_data, y_train)
