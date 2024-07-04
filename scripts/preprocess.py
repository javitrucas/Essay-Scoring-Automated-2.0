import re
import numpy as np
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import nltk
import fasttext
import os
import urllib.request
import zipfile

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove digits
    text = re.sub(r'\b\d+\b', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Lemmatize words and remove stop words
    text = ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word not in stop_words])
    return text

# Function to perform synonym replacement for data augmentation
def synonym_replacement(words, n):
    new_words = words.copy()
    # Get a list of random words from the input excluding stop words
    random_word_list = list(set([word for word in words if word not in stop_words]))
    np.random.shuffle(random_word_list)
    num_replaced = 0
    # Replace random words with their synonyms
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = np.random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

# Function to get synonyms of a word using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# Read train and test datasets
train_df = pd.read_csv('/home/javitrucas/essay_scoring/data/train.csv')
test_df = pd.read_csv('/home/javitrucas/essay_scoring/data/test_prueba.csv')

# Check for null values in the datasets
print("None values in train_df:", train_df.isnull().values.any())
print("None values in test_df:", test_df.isnull().values.any())

# Clean the text data in the train dataset
train_df['cleaned_text'] = train_df['full_text'].apply(clean_text)
print("None values after text cleaning:", train_df['cleaned_text'].isnull().values.any())
print("Null values in train_df['score']:", train_df['score'].isnull().values.any())
train_df.dropna(subset=['cleaned_text', 'score'], inplace=True)

# Balance the dataset by upsampling the minority class
train_df_majority = train_df[train_df['score'] == train_df['score'].mode()[0]]
train_df_minority = train_df[train_df['score'] != train_df['score'].mode()[0]]
train_df_minority_upsampled = resample(train_df_minority, 
                                      replace=True,    
                                      n_samples=len(train_df_majority),
                                      random_state=42)
train_df_balanced = pd.concat([train_df_majority, train_df_minority_upsampled])

# Save the balanced dataset
train_df_balanced.to_csv('/home/javitrucas/essay_scoring/data/train_balanced.csv', index=False)

# Save cleaned sentences to a text file for FastText training
sentences = train_df_balanced['cleaned_text'].tolist()
with open('/home/javitrucas/essay_scoring/data/cleaned_sentences.txt', 'w') as f:
    for sentence in sentences:
        f.write("%s\n" % sentence)

# Train a FastText model on the cleaned sentences
model = fasttext.train_unsupervised('/home/javitrucas/essay_scoring/data/cleaned_sentences.txt', model='skipgram', dim=100)
model.save_model('/home/javitrucas/essay_scoring/data/word2vec_fasttext.bin')

# Download the GloVe embeddings
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_zip_path = "/home/javitrucas/essay_scoring/data/glove.6B.zip"
glove_extract_path = "/home/javitrucas/essay_scoring/data/"

if not os.path.exists(glove_zip_path):
    print("Downloading GloVe embeddings...")
    urllib.request.urlretrieve(glove_url, glove_zip_path)
    print("Download completed.")

# Extract the GloVe embeddings
if not os.path.exists(os.path.join(glove_extract_path, "glove.6B.100d.txt")):
    print("Extracting GloVe embeddings...")
    with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
        zip_ref.extractall(glove_extract_path)
    print("Extraction completed.")

# Verify the GloVe file exists after extraction
assert os.path.exists(os.path.join(glove_extract_path, "glove.6B.100d.txt")), "GloVe file not found after extraction."

# Load the GloVe embeddings into a dictionary
embedding_index = {}
with open('/home/javitrucas/essay_scoring/data/glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Prepare the data for training
X = train_df_balanced['cleaned_text'].values
y = train_df_balanced['score'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
maxlen = 300
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen)

# Print shapes of the padded sequences
print(f'X_train_pad shape: {X_train_pad.shape}')
print(f'X_val_pad shape: {X_val_pad.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')
print("None values in X_train_pad:", np.any(X_train_pad == None))
print("None values in X_val_pad:", np.any(X_val_pad == None))

# Save the processed data to .npy files
np.save('/home/javitrucas/essay_scoring/data/X_train_pad.npy', X_train_pad)
np.save('/home/javitrucas/essay_scoring/data/X_val_pad.npy', X_val_pad)
np.save('/home/javitrucas/essay_scoring/data/y_train.npy', y_train)
np.save('/home/javitrucas/essay_scoring/data/y_val.npy', y_val)
np.save('/home/javitrucas/essay_scoring/data/essay_texts.npy', X_val)

# Save the tokenizer to a JSON file
tokenizer_json = tokenizer.to_json()
with open('/home/javitrucas/essay_scoring/data/tokenizer.json', 'w') as f:
    f.write(tokenizer_json)
