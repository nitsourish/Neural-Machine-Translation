
# coding: utf-8

# In[1]:

import pandas as pd
import os as os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import datetime
from datetime import date,timedelta
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import re,unicodedata
import string

os.chdir('C:\practice\Machine_translation\deu-eng')
os.getcwd()
# load doc into memory
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text
filename = 'deu.txt'

 #Split by lines and pairs
def split_pair(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t')for line in lines]
    return pairs
     
#Cleaning text
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = unicodedata.normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return np.array(cleaned)
        
#function save_clean_data() uses the pickle API to save the list of clean text to file.
def save_clean_data(sentence,filename):
    dump(sentence,open(filename,'wb'))
    print('Saved:%s'% filename)


# In[2]:

#execute all these function
doc = load_doc(filename)
pairs = split_pair(doc)


# In[3]:

clean_pairs = clean_pairs(pairs)


# In[4]:

# spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))


# In[5]:

from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
 
# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('english-german.pkl')
 
# reduce dataset size
n_sentences = 15000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:14000], dataset[14000:]
# save
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')


# In[6]:

# find maximum length of sentence- tokenizer
def token(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return(tokenizer)

def max_len(lines):
    return max(len(line.split())for line in lines)


# In[7]:

def load_clean_sentences(filename):
    return(load(open(filename, 'rb')))
#load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')


# In[8]:

from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
eng_tokenizer = token(lines=dataset[:,0])
eng_vocab = len(eng_tokenizer.word_index)+1
eng_max = max_len(dataset[:,0])
ger_tokenizer = token(lines=dataset[:,1])
ger_vocab = len(ger_tokenizer.word_index)+1
ger_max = max_len(dataset[:,1])
print('english vocab size: %d' % eng_vocab)
print('german vocab size: %d' % ger_vocab)
print('english max_len size: %d' % eng_max)
print('german max_len: %d' % ger_max)


# In[9]:

#data prep for model
#sequence,padding of X and one hot encoding of y
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer 
def encode_text(tokenizer,max_len,lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X,maxlen=max_len,padding='post')
    return X

#One hot encoding of output
def one_hot_encode(sequences, vocab_size):
    Y=[]
    for sequence in sequences:
        y_seq = to_categorical(sequence,num_classes=vocab_size)
        Y.append(y_seq)
    Y=np.array(Y)    
    Y=Y.reshape(sequences.shape[0],sequences.shape[1],vocab_size)  
    return Y


# In[10]:

#train-data
train_X = encode_text(ger_tokenizer,ger_max,train[:,1])
#train_X = one_hot_encode(train_X,vocab_size=ger_vocab)
train_Y=encode_text(eng_tokenizer,eng_max,train[:,0])
train_Y=one_hot_encode(train_Y,vocab_size=eng_vocab)

#test-data
test_X = encode_text(ger_tokenizer,ger_max,test[:,1])
#test_X=one_hot_encode(test_X,vocab_size=ger_vocab)
test_Y=encode_text(eng_tokenizer,eng_max,test[:,0])
test_Y=one_hot_encode(test_Y,vocab_size=eng_vocab)


# In[23]:

#Keras sequential model-Encoder-decoder
import keras
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda,TimeDistributed
from keras.optimizers import Adam
from keras.utils import to_categorical,plot_model
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import pydot
import graphviz
from keras.callbacks import ModelCheckpoint

def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model
 
# define model
model = define_model(ger_vocab, eng_vocab, ger_max, eng_max, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
#plot_model(model, to_file='model.png', show_shapes=True)
model.compile(loss='categorical_crossentropy',optimizer='adam')
print(model.summary())
#fit model
# checkpoint
filepath='model_h5'
checkpoint = ModelCheckpoint(filepath,mode='max',monitor='val_acc',save_best_only=True,verbose=1)
model.fit(train_X,train_Y,batch_size=64,callbacks=[checkpoint],epochs=30,validation_data=[test_X,test_Y])


# In[23]:

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# In[41]:

predicted


# In[28]:

from nltk.translate.bleu_score import corpus_bleu
# Model performance
#integer to word#
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [np.argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
 
# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, eng_tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % corpus_bleu([actual], [predicted], weights=(0.25, 0.25, 0.25, 0.25))) 


# In[33]:

#implement functions
#evaluate_model(model,eng_tokenizer,train_X,train)
evaluate_model(model,eng_tokenizer,train_X,train)
# calculate BLEU score
#from nltk.translate.bleu_score import corpus_bleu
#print('BLEU-1: %f' % corpus_bleu(actual[1], predicted[1], weights=(1.0, 0, 0, 0)))
#print('BLEU-2: %f' % corpus_bleu(actual[1], predicted[1], weights=(0.5, 0.5, 0, 0)))
#print('BLEU-3: %f' % corpus_bleu(actual[1], predicted[1], weights=(0.3, 0.3, 0.3, 0)))
#print('BLEU-4: %f' % corpus_bleu(actual[1], predicted[1], weights=(0.25, 0.25, 0.25, 0.25)))


# In[27]:

# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0.0))
print(score)


# In[126]:

from nltk import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
reference = actual
hypothesis = predicted
smoothie = SmoothingFunction().method4
print('bleu_score.corpus_bleu(reference, hypothesis): {0}'.
      format(bleu_score.corpus_bleu(reference,hypothesis,smoothing_function=smoothie)))


# In[1]:

import nltk
import pkg_resources
pkg_resources.get_distribution("nltk").version


# In[15]:

#Attention Model

#train-data
train_X = encode_text(ger_tokenizer,ger_max,train[:,1])
train_X = one_hot_encode(train_X,vocab_size=ger_vocab)
train_Y=encode_text(eng_tokenizer,eng_max,train[:,0])
train_Y=one_hot_encode(train_Y,vocab_size=eng_vocab)

#test-data
test_X = encode_text(ger_tokenizer,ger_max,test[:,1])
test_X = one_hot_encode(test_X,vocab_size=ger_vocab)
test_Y=encode_text(eng_tokenizer,eng_max,test[:,0])
test_Y=one_hot_encode(test_Y,vocab_size=eng_vocab)


# In[25]:

Tx=train_X.shape[1]
m=train_X.shape[0]
print(Tx)
Ty=train_Y.shape[1]
m


# In[26]:

repeator = RepeatVector(Tx)  
concatenator = Concatenate(axis=-1)  
densor1 = Dense(100, activation = "tanh")  
densor2 = Dense(1, activation = "relu")  
activator = Activation('softmax', name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook  
dotor = Dot(axes = 1) 


# In[27]:

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

#from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
#from nmt_utils import *
import matplotlib.pyplot as plt

def one_step_attention(a, s_prev):  
    """ 
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights 
    "alphas" and the hidden states "a" of the Bi-LSTM. 
     
    Arguments: 
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a) 
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s) 
     
    Returns: 
    context -- context vector, input of the next (post-attetion) LSTM cell 
    """  
      
    ### START CODE HERE ###  
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)  
    s_prev = repeator(s_prev)  
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)  
    concat = concatenator([a,s_prev])  
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)  
    e = densor1(concat)  
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)  
    energies = densor2(e)  
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)  
    alphas = activator(energies)  
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)  
    context = dotor([alphas,a])  
    ### END CODE HERE ###  
      
    return context 


# In[28]:

#Model define:
import gensim
def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    from keras import regularizers
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    ### START CODE HERE ###
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    
    a = Bidirectional(LSTM(n_a,return_sequences=True,dropout=0.3, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros'),input_shape=(m,Tx,2*n_a))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a,s)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = LSTM(n_s,return_state=True,dropout=0.3, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros')(context,initial_state = [s, c])
        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        #out = Dense(256,activation='tanh')(s)
        out = Dense(machine_vocab_size,activation='softmax')(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X,s0,c0],outputs=outputs)
    
    ### END CODE HERE ###
    
    return model


# In[29]:

model = model(Tx=ger_max,Ty=eng_max,n_a=64,n_s=128,human_vocab_size=ger_vocab,machine_vocab_size=eng_vocab)
model.summary()


# In[30]:

#Model compile and fit
from keras.callbacks import EarlyStopping,ModelCheckpoint
opt = Adam(beta_1=0.9,beta_2=0.999,lr=0.03,decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
s0 = np.zeros([m,128])
c0 = np.zeros([m,128])
output = list(train_Y.swapaxes(0,1))
model.fit(x=[train_X,s0,c0], y=output,epochs=3,batch_size=128,verbose=1,validation_split=0.05,callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)])


# In[31]:

# serialize model to JSON
model_json = model.to_json()
with open("attention_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("attention_model.h5")
print("Saved model to disk")


# In[22]:

from nltk.translate.bleu_score import corpus_bleu
# Model performance
#integer to word#
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)
	integers = [np.argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
 
# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        source = source.reshape((1, source.shape[0],source.shape[1]))
        translation = predict_sequence(model, eng_tokenizer, [source,s0,c0])
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    #print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    #print('BLEU-4: %f' % corpus_bleu([actual], [predicted], weights=(0.25, 0.25, 0.25, 0.25))) 


# In[23]:

evaluate_model(model,eng_tokenizer,train_X,train)


# In[31]:

n_a = 64
n_s = 128
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(eng_vocab, activation='softmax')


# In[60]:

Sources = [train[1:2,1],train[2:3,1],train[5:6,1],train[12:13,1],train[4:5,1]]
for sourece in Sources:
    source = encode_text(ger_tokenizer,ger_max,source)
    source= one_hot_encode(source,vocab_size=ger_vocab)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output))


# In[40]:

np.array(source,dtype='U291')


# In[70]:

source = train[16:17,1]
source = encode_text(ger_tokenizer,ger_max,source)
source= one_hot_encode(source,vocab_size=ger_vocab)
prediction = model.predict([source, s0, c0])
prediction = np.argmax(prediction, axis = -1)
prediction.shape


# In[59]:

integers = [np.argmax(vector) for vector in prediction]
target = list()
for i in integers:
    word = word_for_id(i, tokenizer)
	target.append(word)
' '.join(target)


# In[87]:

prediction = model.predict([source,s0,c0], verbose=0)
integers = np.argmax(prediction)
integers


# In[107]:

model.predict([source,s0,c0], verbose=0)


# In[128]:

np.argmax(model.predict([train_X[1333:1334,:],s0,c0], verbose=0)[0])


# In[133]:

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)
	integers = [np.argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)


# In[111]:

del embedding_matrix


# In[11]:

#Pretrained embedding-GLOVE
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(dataset[:,1])
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(dataset[:,0])
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = max_len(dataset[:, 0])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)


# In[12]:

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.50d.txt',encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[14]:

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


# In[30]:

#Keras LSTM model with pretrained embeddings
import keras
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda,TimeDistributed
from keras.optimizers import Adam
from keras.utils import to_categorical,plot_model
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import pydot
import graphviz
from keras.callbacks import ModelCheckpoint
from keras import regularizers

def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(src_vocab,50,weights=[embedding_matrix], input_length=src_timesteps, mask_zero=True, trainable = False))
    model.add(Bidirectional(LSTM(n_units)))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model
 
# define model
model = define_model(ger_vocab, eng_vocab, ger_max, eng_max, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
#plot_model(model, to_file='model.png', show_shapes=True)
model.compile(loss='categorical_crossentropy',optimizer='adam')
print(model.summary())
#fit model
# checkpoint
filepath='model_h5'
checkpoint = ModelCheckpoint(filepath,mode='max',monitor='val_acc',save_best_only=True,verbose=1)
model.fit(train_X,train_Y,batch_size=64,callbacks=[checkpoint],epochs=30,validation_data=[test_X,test_Y])


# In[79]:

#Wordtovec embeddings
from gensim.models import Word2Vec
model = Word2Vec(train[:,1].tolist(), size=100, window=5, min_count=5, workers=25, sg=0, negative=5)
word_vectors = model.wv
print("Number of word vectors: {}".format(len(word_vectors.vocab)))


# In[81]:

from collections import Counter
MAX_NB_WORDS = len(word_vectors.vocab)
MAX_SEQUENCE_LENGTH = 200
from keras.preprocessing.sequence import pad_sequences
vocab = Counter()
word_index = {t[0]: i+1 for i,t in enumerate(vocab.most_common(MAX_NB_WORDS))}
sequences = [[word_index.get(t, 0) for t in train[:,1].tolist()]
             for comment in train[:,1].tolist()[:len(train[:,1].tolist())]]
test_sequences = [[word_index.get(t, 0) for t in test[:,1].tolist()]
             for comment in test[:,1].tolist()[:len(test[:,1].tolist())]]

# pad
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, 
                     padding="pre", truncating="post")


test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre",
                          truncating="post")
print('Shape of test_data tensor:', test_data.shape)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, 
                     padding="pre", truncating="post")
print('Shape of data tensor:', data.shape)


# In[83]:

WV_DIM = 100
nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
# we initialize the matrix with random numbers
wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization    


# In[96]:

wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)

# Inputs
comment_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = wv_layer(comment_input)

# biGRU
embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)


# In[100]:

# Output
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
preds = Dense(eng_vocab, activation='sigmoid')(x)
# build the model
model = Model(inputs=[train_X], outputs=train_Y)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
              metrics=[])


# In[102]:

model = Model(inputs=[train_X], outputs=train_Y)


# In[113]:

train_X = one_hot_encode(train_X,vocab_size=ger_vocab)

