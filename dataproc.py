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


if __name__ == '__main__':
	getid_line_mapping()
