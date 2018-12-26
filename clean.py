pos = ['NOUN','ADV']

with open(r"C:\Users\rauna\OneDrive\Documents\GitHub\mraunak-rvijan-pnegandh-a3\part1\bc.train") as f:
    txt_inp = f.read().split('\n')
    for sentence in txt_inp:
    	count  = 1
    	for word in sentence.split():
    		if(count%2):
    			print(word, end = "")
    		count+=1
    	print()