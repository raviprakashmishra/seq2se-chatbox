separator = " +++$+++ "

limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
        }

UNK = 'unk'
VOCAB_SIZE = 8000

import random

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

def index_(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

if __name__ == '__main__':
	id2line = getid_line_mapping()
	convs = get_conversations()
	print (convs)
	create_conversations(convs,id2line)
	q,a = gather_dataset(convs,id2line)
	prepare_seq2seq_files(q,a)


