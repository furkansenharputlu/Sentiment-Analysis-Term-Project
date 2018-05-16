# -*- coding: utf-8 -*-
#!/usr/bin/python

import string


with open("combined-train",'w') as comb:
	with open("Train/negative-train", 'r') as f:
		text = f.read()
		text = text.translate(str.maketrans('', '', string.punctuation))
		lines = text.replace("\t\t\t","\t").splitlines()
		for line in lines:
			comb.write("-1\t" + line + '\n')
			# comb.write("-1\t" + line + '\n')
	
	with open("Train/notr-train",'r') as f:
		text = f.read()
		text = text.translate(str.maketrans('', '', string.punctuation))
		lines = text.replace("\t\t\t","\t").splitlines()
		for line in lines:
			comb.write("0\t" + line + '\n')
	
	with open("Train/positive-train", 'r') as f:
		text = f.read()
		text = text.translate(str.maketrans('', '', string.punctuation))
		lines = text.replace("\t\t\t","\t").splitlines()
		for line in lines:
			comb.write("1\t" + line + '\n')


