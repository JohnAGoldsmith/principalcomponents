#-*- coding: <utf-16> -*-
#import codecs
import os
import sys
import string
import operator
mywords=dict()
from math import sqrt
from  collections import defaultdict
import numpy as np
from scipy import linalg as LA
import scipy.sparse as spr

from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
#import nimfa

#from scipy.sparse.linalg import svds
#from numpy import *
#import time
#from pyx import *


GraphicsFlag = False





#-----------------------------------------------------------------------#
#                                    #
#    This program takes a trigram file and a word list        #
#    and creates a file with lists of most similar words.        #
#    John Goldsmith and Wang Xiuli 2012.                #
#                                    #
#-----------------------------------------------------------------------#

def from_data_points_to_focus_words_and_contexts(data_points):
        for (focus_word, context, this_count) in data_points:
                if context_usage[context] < minimum_context_use:
                    continue
                if simple_word_count_in_corpus[focus_word] < minimum_word_use:
                    continue
                if context not in context_to_index:
                    index_value = len(context_to_index)
                    context_to_index[context] = index_value
                    index_to_context[index_value] = context
                    #print 189, context, context_usage[context]
                if focus_word not in focus_word_to_index:
                    index_value = len(focus_word_to_index)
                    focus_word_to_index[focus_word] = index_value
                    index_to_focus_word[index_value] = focus_word
                if focus_word not in focus_word_usage:
                    focus_word_usage[focus_word] = 0
                focus_word_usage[focus_word] += 1
                    #print focus_word
                    
                #print context, context_usage[context], 193
                   
        context_count = len(context_to_index)
        focus_word_count = len(focus_word_to_index)
        #print 62, context_count, focus_word_count
        return context_count, focus_word_count
        
        
        
        

def from_data_points_list_to_data_array(data,datapoints):     
        #print 227, focus_word_count, context_count
        for word,context,count in data_points:
            if context not in context_to_index:
                continue
            if word not in focus_word_to_index:
                continue
            #print word, context
            data[focus_word_to_index[word],context_to_index[context]] = 1 #bad results when I put "count" there
            #print word, context, 214
        print "   Established data array."    
        print "   outfile folder", outfilename_word_to_contexts


#def print_contexts(outfileContexts, data):
 
            
def print_contexts_to_words(outfile,data):   
        contexts = context_to_index.keys()
        contexts.sort()
        for context  in contexts:
            temp_word_list = list()        
            print >>outfile, context
            for word in focus_word_to_index:
                if data[focus_word_to_index[word], context_to_index[context]] > 0:
#                        print >>outfile, "\t", word,  data[focus_word_to_index[word], context_to_index[context]]
                        temp_word_list.append(word)
                temp_word_list.sort()
            for word in temp_word_list:
                print >>outfile, "\t", word, data[focus_word_to_index[word], context_to_index[context]]
            print >>outfile

def print_words_to_contexts(outfile, data):
        temp_words = focus_word_to_index.keys()
        temp_words.sort()
        for word  in temp_words:
            templist = list()
            print >>outfile, word
            for context in context_to_index:
                if data[focus_word_to_index[word], context_to_index[context]] > 0:
                        templist.append(context)
            templist.sort()
            for context in templist:
                  print >>outfile, "\t", context,  data[focus_word_to_index[word], context_to_index[context]]
            print >>outfile

         

def print_eigenvectors_of_contexts_to_file(outfileEigenvectors, myeigenvectors, myeigenvalues):
        print ("---Printing contexts to latex file.")
        formatstr1 = '%20d  %15s %10.3f %5i'
        headerformatstr = '%20s  %15s %10.3f '
        print "   Printing to ", outfolder
        print "   Number of eigenvectors:", NumberOfEigenvectors
        print "   Number of contexts", context_count
        if PrintEigenvectorsFlag:
            for eigenno in range(NumberOfEigenvectors):
                #print "  eigenno", eigenno
                print >>outfileEigenvectors
                print >>outfileEigenvectors,headerformatstr %("Eigenvector number", "context", myeigenvalues[eigenno])
                print >>outfileEigenvectors,"_____________________________________________"
                templist = list()        
                for contextno in range(context_count):
                    templist.append((contextno, myeigenvectors[contextno,eigenno])) 
                    #print 347, contextno, "context no"
                templist.sort(key = lambda second : second[1])
                for i in range(context_count):
                    mypair = templist[i]
                    contextno = mypair[0]
                    eigenvalue = mypair[1]
                    context = index_to_context[contextno]
                    print >>outfileEigenvectors, formatstr1 %(eigenno, context, eigenvalue, context_usage[context])
                    #print formatstr %(eigenno, context, eigenvalue)
            
def NNMF():
    #V = spr.csr_matrix([[1, 0, 2, 4], [0, 0, 6, 3], [4, 0, 5, 6]])
    #print('Target:\n%s' % data.todense())

    nmf = nimfa.Nmf(data, max_iter=200, rank=2, update='euclidean', objective='fro')
    nmf_fit = nmf()
    
    W = nmf_fit.basis()
    print('Basis matrix:\n%s' % W)
    
    H = nmf_fit.coef()
    print('Mixture matrix:\n%s' % H)
    
    print('Euclidean distance: %5.3f' % nmf_fit.distance(metric='euclidean'))
    
    sm = nmf_fit.summary()
    print('Sparseness Basis: %5.3f  Mixture: %5.3f' % (sm['sparseness'][0], sm['sparseness'][1]))
    print('Iterations: %d' % sm['n_iter'])
    #print('Target estimate:\n%s' % np.dot(W.todense(), H.todense()))
         
#---------------------------------------------------------------------------#
#    Variables to be changed by user
#---------------------------------------------------------------------------#
LatexFlag = True
PrintEigenvectorsFlag = True
unicodeFlag = False
FileEncoding =  "ascii"

datafilelocation     =  "../../../data/"
languagename     = "english-browncorpus"
languagename = "english-encarta"

wordfolder = datafilelocation + languagename + "/dx1/"
trigramfolder = datafilelocation + languagename + "/ngrams/"
outfolder = datafilelocation + languagename + "/neighbors/"

outfile_eps         = outfolder + "dx2_files/results/" + languagename + ".eps"
outfile_pdf         = outfolder + "dx2_files/results/" + languagename + ".pdf"

#NumberOfWordsForContext     = 1000 # 40000
NumberOfEigenvectors         = 20
NumberOfWordsForAnalysis    = 500 #4000
NumberOfNeighbors          = 9

punctuation         = " $/+.,;:?!()\"[];        "



#---------------------------------------------------------------------------#
#    File names
#---------------------------------------------------------------------------#


infileTrigramsname = trigramfolder + languagename + "_trigrams.txt"
infileWordsname = wordfolder + languagename + ".dx1"
outfilename_word_eigenvectors = outfolder + languagename + "_word_eigenvectors" + ".txt"
outfilename_context_eigenvectors = outfolder + languagename + "_context_eigenvectors" + ".txt"
outfilenameNeighbors = outfolder + languagename + "_PoS_closest" + "_" + str(NumberOfNeighbors ) + "_neighbors.txt"
outfilenameLatex = outfolder + languagename + "_latex.tex"
outfilename_word_to_contexts = outfolder + languagename + "_focus_word_to_contexts.txt"
outfilename_context_to_words = outfolder + languagename + "_context_to_words.txt"
outfilenameEigenCoordinates = outfolder + languagename + "_eigen_coordinates.txt"

print ("\n\nI am looking for a trigrams file name: ", infileTrigramsname)

#---------------------------------------------------------------------------#
#    Variables
#---------------------------------------------------------------------------#


#analyzedwordlist = list() # this means that the info comes from the independent word file


#---------------------------------------------------------------------------#
#    Open files for reading and writing
#---------------------------------------------------------------------------#

if unicodeFlag:
    trigramfile         =codecs.open(infileTrigramsname, encoding = FileEncoding)
    wordfile         =codecs.open(infileWordsname, encoding = FileEncoding)
    if PrintEigenvectorsFlag:
        outfileEigenvectors = codecs.open (outfilename1, "w",encoding = FileEncoding)
    outfileNeighbors    = codecs.open (outfileneighborsname, "w",encoding = FileEncoding)

else:
    outfile_word_eigenvectors = open (outfilename_word_eigenvectors, "w")
    outfile_context_eigenvectors = open (outfilename_context_eigenvectors, "w")
    outfileNeighbors    = open (outfilenameNeighbors, "w")
    outfileLatex         = open (outfilenameLatex, "w")
    outfile_words_to_contexts     = open (outfilename_word_to_contexts, "w")
    outfile_context_to_words = open (outfilename_context_to_words, "w")
    eigencoordinates     = open (outfilenameEigenCoordinates, "w")

    wordfile        = open(infileWordsname)
    trigramfile         = open(infileTrigramsname)

print "Language is", languagename +"."
print "File name:", languagename+ "."

print >>outfile_word_eigenvectors,"#", \
            languagename, "\n#", \
            "Number of words analyzed", NumberOfWordsForAnalysis,"\n#", \
            "Number of neighbors identified", NumberOfNeighbors, "\n#","\n#"
print >>outfile_context_eigenvectors,"#", \
            languagename, "\n#", \
            "Number of words analyzed", NumberOfWordsForAnalysis,"\n#", \
            "Number of neighbors identified", NumberOfNeighbors, "\n#","\n#"

print >>outfileNeighbors, "#", \
        languagename, "\n#",\
        "Number of words analyzed", NumberOfWordsForAnalysis,"\n#", \
        "Number of neighbors identified", NumberOfNeighbors,"\n#","\n#"


#---------------------------------------------------------------------------#
#    Read trigram file
#---------------------------------------------------------------------------#
total_word_count = 0
simple_word_count_in_corpus = dict()
focus_word_to_index = dict()
index_to_focus_word = dict()
context_to_index = dict()
index_to_context = dict()
context_usage = dict()
datapoints = np.zeros(( ))

#minimum_number_of_words_in_each_context = 3  #aka context_usage
minimum_context_use = 100
minimum_word_use = 200

if True:


    print "1. Reading in trigram file."
    data_count = 0
    trigram_list = list()
    data_points = list()
    for line in trigramfile:
        line = line.split()
        if line[0] == "#":
            continue
        context1 = line[0]
        context2 = line[2]
        punctuation = ",.()_"
        if context1 in punctuation or context2 in punctuation:
            continue
        focus_word = line[1]
        if focus_word in punctuation:
            continue
        this_count = int(line[3])
        context = line[0] + " _ " +  line[2]
        if not context in context_usage:
            context_usage[context] = 0
        context_usage[context] += 1
        data_points.append((focus_word,context,this_count ))
        if focus_word not in simple_word_count_in_corpus:
            simple_word_count_in_corpus[focus_word] = 1
        else:
            simple_word_count_in_corpus[focus_word] += 1
        
    print "   Number of trigrams obtained: ", len(data_points)

word_index = np.zeros((len(data_points)))
context_index = np.zeros((len(data_points)))
focus_word_usage = dict()
context_count = 0
focus_word_count = 0

#------------------------------------------
context_count, focus_word_count = from_data_points_to_focus_words_and_contexts(data_points)
#------------------------------------------

if context_count < NumberOfEigenvectors:
    NumberOfEigenvectors = context_count
    print "   Number of eigenvectors changed to ", NumberOfEigenvectors
 
print "   Minimum context usage:", minimum_context_use
print "   Focus word count (types) = ", focus_word_count
print "   Context count (types) = ", context_count

data = np.zeros((focus_word_count, context_count ))
from_data_points_list_to_data_array(data,datapoints)

#------------------------------------------
#print_contexts(outfile_word_to_contexts, data)
print_contexts_to_words(outfile_context_to_words,data)
print_words_to_contexts(outfile_words_to_contexts,data)
#------------------------------------------

np.set_printoptions(edgeitems=15)
        
#---------------------------------------------------------------------------#
#    Make zero-mean
#---------------------------------------------------------------------------#

data1 = np.zeros((focus_word_count, context_count))
data2 = np.zeros((focus_word_count, context_count))
normal_PCA = True 
if (normal_PCA):
        #for contexts
        #print 226, data
        data1 -= data - np.mean(data,axis=0)

          
        # for words
        data2 = np.transpose(data)
        #print data2
        data2 -= np.mean(data2,axis=0)
        #print data2
print "  Established data array."    
#---------------------------------------------------------------------------#
#    Non-negative matrix factorization.
#---------------------------------------------------------------------------#

M = spr.csr_matrix(data)
NNMF = False
#V=np.zeros((focus_word_count, context_count))
#---------------------------------------------------------------------------#
if NNMF:
        nmf = nimfa.Nmf(data, seed=None, rank=10, max_iter=12, update='euclidean',
                objective='fro')
#---------------------------------------------------------------------------#




#---------------------------------------------------------------------------#
#    Compute eigenvectors.
#---------------------------------------------------------------------------#
print   "2. Compute eigenvectors for contexts.",
# Contexts...

if (False):
    mycovar = np.matmul(np.transpose(data1),  data1)
    myeigenvalues, myeigenvectors = np.linalg.eigh(mycovar)
    print "Done."


    
if normal_PCA:  # regular pca, not non-negative...    
    n, m = data1.shape
    assert np.allclose(data1.mean(axis=0), np.zeros(m))
    C = np.dot(data1.T, data1) / (n-1) #this computes the covariance matrix
    myeigenvalues, myeigenvectors = np.linalg.eig(C)
    # Project X onto PC space
    X_pca = np.dot(data1, myeigenvectors)
    #return X_pca

print_eigenvectors_of_contexts_to_file(outfile_context_eigenvectors, myeigenvectors, myeigenvalues)


NNMF_nimfa = False
if NNMF_nimfa:
    NNMF()
        
NNMF_sklearn = False
if NNMF_sklearn == True:
    
    nmf = NMF( random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(data)
    
    model = NMF()
    W = model.fit_transform(data)
    H = model.components_
   
    

#---------------------------------------------------------------------------#
# Focus words...
print   "2. Compute eigenvectors for words..."

mycovar2 = np.matmul(data, np.transpose(data))
myeigenvalues2, myeigenvectors2 = np.linalg.eigh(mycovar2)  # compute eigenvectors


print "  Eigenvectors for words complete."

n, m = data2.shape
#print 342, data2
assert np.allclose(data2.mean(axis=0), np.zeros(m))

Covariance_matrix = np.dot(data2.T, data2) / (n-1) 
myeigenvalues2, myeigenvectors2 = np.linalg.eig(Covariance_matrix)
print "  Eigen computation finished."


# Project X onto PC space
#X_pca = np.dot(X, eigen_vecs)
#return X_pca
print "   Done."
 
print ("10. Printing words to latex file.")
formatstr = '%20d  %15s %10.3f %10i'
headerformatstr = '%20s  %15s %10.3f'
print "   Printing to ", outfolder

if PrintEigenvectorsFlag:
    for eigenno in range(NumberOfEigenvectors):
        print >>outfile_word_eigenvectors
        print >>outfile_word_eigenvectors,headerformatstr %("Eigenvector number", "word" , myeigenvalues2[eigenno])
        print >>outfile_word_eigenvectors,"_____________________________________________"
        templist = list()        
        for wordno in range(focus_word_count):
	    #print wordno, index_to_focus_word[wordno]
            templist.append((wordno, myeigenvectors2[wordno,eigenno]))            
        templist.sort(key = lambda second : second[1])
        for i in range(focus_word_count):
            mypair = templist[i]
            wordno = mypair[0]
            eigenvalue = mypair[1]
            word = index_to_focus_word[wordno]
            print >>outfile_word_eigenvectors, formatstr %(eigenno, word, eigenvalue, focus_word_usage[word])
            #print eigenno, word, eigenvalue
#---------------------------------------------------------------------------#


outfile_word_eigenvectors.close()
outfile_context_eigenvectors.close()


if LatexFlag:
    #Latex output
    print >>outfileLatex, "%",  infileWordsname
    print >>outfileLatex, "\\documentclass{article}"
    print >>outfileLatex, "\\usepackage{booktabs}"
    print >>outfileLatex, "\\begin{document}"

data = dict() # key is eigennumber, value is list of triples: (index, word, eigen^{th} coordinate) sorted by increasing coordinate
print ("9. Printing contexts to latex file.")
formatstr = '%20d  %15s %10.3f'
headerformatstr = '%20s  %15s %10.3f'
NumberOfWordsToDisplayForEachEigenvector = 20

# 2018

# PUT THIS BACK IN
#g = graph.graphxy(width=8)
coordinates = list()





#PUT THESE BACK IN
#g.plot(graph.data.points(coordinates,x=1,y=2))
#g.writeEPSfile(outfile_eps)
#g.writePDFfile(outfile_pdf)
#outfile_eps.close()
#outfile_pdf.close()
 
LatexFlag = False
if LatexFlag:
    for eigenno in range(NumberOfEigenvectors):
        eigenlist=list()
        #data[eigenno] = list()
        data = list()
        #for wordno in range (NumberOfWordsForAnalysis):
        #    eigenlist.append( (wordno,myeigenvectors[wordno, eigenno]) )
        eigenlist.sort(key=lambda x:x[1])
        print >>outfileLatex
        print >>outfileLatex, "Eigenvector number", eigenno, "\n"
        print >>outfileLatex, "\\begin{tabular}{lll}\\toprule"
        print >>outfileLatex, " & word & coordinate \\\\ \\midrule "

 
        print >>outfileLatex, "\\bottomrule \n \\end{tabular}", "\n\n"
        print >>outfileLatex, "\\newpage"
print >>outfileLatex, "\\end{document}"

#---------------------------------------------------------------------------#
#     Print contexts shared by nearby words
#---------------------------------------------------------------------------#

print "Exiting successfully."

outfile_context_eigenvectors.close()
outfile_word_eigenvectors.close()
outfileNeighbors.close()




