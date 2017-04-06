separator = " +++$+++ "

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

if __name__ == '__main__':
	id2line = getid_line_mapping()
	convs = get_conversations()
	print (convs)
	create_conversations(convs,id2line)


