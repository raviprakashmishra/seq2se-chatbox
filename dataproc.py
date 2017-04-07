separator = " +++$+++ "
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
        }

UNK = 'unk'
VOCAB_SIZE = 8000

import random

import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle

def getid_line_mapping():
	
	lines=open('raw_data/movie_lines.txt', encoding='utf-8', errors='ignore').read()
	#print ("printing lines",lines)
	lines=lines.split('\n')
	#print (lines)
	id2line = {}
	for line in lines:
		split = line.split(separator)
		if len(split) == 5:
			id2line[split[0]] = split[4]
	#print (id2line)

	return id2line

def get_conversations():
    conv_lines = open('raw_data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    #print (conv_lines)
    # exclude possible empty last character
    conv_lines = conv_lines[:-1]
    convs = [ ]
    for line in conv_lines:
    	split = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    	convs.append(split.split(','))
    print ("printing conversations", convs)
    return convs

'''
read convs created above
and create and save each convs in 
a new text file

'''
def create_conversations(convs,id2line,path=""):
	f_id = 0
	for conv in convs:
		 f_conv = open(path + str(f_id)+'.txt', 'w')
		 for id in conv:
		 	f_conv.write(id2line[id])
		 	f_conv.write('\n')
		 f_conv.close()
		 f_id +=1


def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        if len(conv) %2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers

'''
prepare file training, test -- encoder
and decoder files
'''
def prepare_seq2seq_files(questions, answers, path='',TESTSET_SIZE = 30000):
	train_enc = open(path + 'train.enc','w')
	train_dec = open(path + 'train.dec','w')
	test_enc  = open(path + 'test.enc', 'w')
	test_dec  = open(path + 'test.dec', 'w')

	test_ids = random.sample(range(0,len(questions)),TESTSET_SIZE)

	for i in range(len(questions)):
		if i in test_ids:
			test_enc.write(questions[i]+'\n')
			test_dec.write(answers[i]+ '\n' )
		else:
			train_enc.write(questions[i]+'\n')
			train_dec.write(answers[i]+ '\n' )
		if i%10000 == 0:
			print('\n>> written {} lines'.format(i))

    # close files
	train_enc.close()
	train_dec.close()
	test_enc.close()
	test_dec.close()

'''
filter questions and answers
which are not in define range
'''

def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    return filtered_q, filtered_a

def index_data(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    

    return filtered_q, filtered_a





def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

     
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

if __name__ == '__main__':
	id2line = getid_line_mapping()
	convs = get_conversations()
	print (convs)
	create_conversations(convs,id2line)
	q,a = gather_dataset(convs,id2line)
	#prepare_seq2seq_files(q,a)
	q = [line.lower() for line in q]
	a = [line.lower() for line in a]

	questions = [ filter_line(line, EN_WHITELIST) for line in q ]
	answers = [ filter_line(line, EN_WHITELIST) for line in a ]

	qlines, alines = filter_data(questions, answers)

	qtokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in qlines ]
	atokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in alines ]

	idx2w, w2idx, freq_dist = index_data( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

	qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)
	idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)
	np.save('idx_q.npy', idx_q)
	np.save('idx_a.npy', idx_a)

	metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
	with open('metadata.pkl', 'wb') as f:
		pickle.dump(metadata, f)

