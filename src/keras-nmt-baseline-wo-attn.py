# Executed Here: https://colab.research.google.com/drive/1FZ-IeeQOKNVYzrUuoTlcXBQA2mICc0j5

import scipy
import numpy as np
from numpy import array
from numpy import argmax
import matplotlib
import pandas
import statsmodels
import sklearn

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

from nltk.translate.bleu_score import corpus_bleu

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n\n')
	pairs = [line.split('\n') for line in  lines]
	return pairs

def switch_pairs(pairs):
      Spairs = []
  for pair in pairs:
    tempPair = [pair[1], pair[0]]
    Spairs.append(tempPair)
  return Spairs

filename = '../data/squad_queries.txt'
#filename = "squad_queries.txt"
#load doc
doc = load_doc(filename)
# split into pairs
pairs = to_pairs(doc)
# switching pairs
Spairs = switch_pairs(pairs)

#array1 = np.asarray(Spairs[:25000])
#array2 = np.asarray(Spairs[25000:50000])
#array3 = np.asarray(Spairs[50000:75000])
#array4 = np.asarray(Spairs[75000:100000])


#dataset = np.concatenate((array1, array2), axis=0)
#dataset = np.concatenate((tempArray, array3), axis=0)
#dataset = np.concatenate((tempArray1, array4), axis=0)


#dataset = np.asarray(Spairs[:10])

dataset = array(Spairs[:1000])


datasetSize = len(dataset)
trainSize = int(datasetSize*0.8)

#splitting intp train and testing 
train, test = dataset[:trainSize], dataset[trainSize:]

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# prepare query tokenizer
query_tokenizer = create_tokenizer(dataset[:,0])
query_vocab_size = len(query_tokenizer.word_index) + 1
query_length = max_length(dataset[:,0])
print('Query Vocabulary Size: %d' % query_vocab_size)
print('Query Max Length: %d' % (query_length))

# prepare question tokenizer
question_tokenizer = create_tokenizer(dataset[:,1])
question_vocab_size = len(question_tokenizer.word_index) + 1
question_length = max_length(dataset[:,1])
print('question Vocabulary Size: %d' % question_vocab_size)
print('question Max Length: %d' % (question_length))

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

trainX = encode_sequences(question_tokenizer, question_length, train[:, 1])

trainY = encode_sequences(query_tokenizer, query_length, train[:, 0])
trainY = encode_output(trainY, query_vocab_size)

testX = encode_sequences(question_tokenizer, question_length, test[:, 1])
testY = encode_sequences(query_tokenizer, query_length, test[:, 0])
testY = encode_output(testY, query_vocab_size)

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences= True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# define model
model = define_model(question_vocab_size, query_vocab_size, question_length, query_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# summarize defined model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)

def word_for_id(integer, tokenizer):
    	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
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
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, query_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s]\ntarget=[%s]\npredicted=[%s] \n \n' % (raw_src, raw_target, translation))
		actual.append([raw_target.split()])
		predicted.append(translation.split())
	
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
 
 # test on some test sequences
print('******* Test. ************')
print
evaluate_model(model, query_tokenizer, testX, test)

