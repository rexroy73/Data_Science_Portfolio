# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 18:40:21 2018

@author: Prashita
"""
import nltk
from nltk import word_tokenize
import numpy as np
import PyPDF2 as pp

#reading file
pdffile = open('JavaBasics-notes.pdf', 'rb')
read_file = pp.PdfFileReader(pdffile)

#getting number of pages
num_pages = read_file.numPages
page_num = range(1,num_pages)
all_pdftext = ''

#getting all text from pdf
for a in page_num:
    page = read_file.getPage(a)
    page_content = page.extractText()
    all_pdftext = all_pdftext + page_content

#print(all_pdftext)
all_pdftextcopy = [all_pdftext]

##########extracting keywords###########

#removing stopwords and words less than 3 letters
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

all_pdftext= all_pdftext.lower()
tokens = word_tokenize(all_pdftext)
tokens = [t for t in tokens if len(t) > 2]
tokens = [t for t in tokens if t not in stopwords]

#getting frequency count of tokens 
freq = nltk.FreqDist(tokens)
for key,val in freq.items():
    print (str(key) + ':' + str(val))

#top words frequency plot
freq.plot(25, cumulative=False)













