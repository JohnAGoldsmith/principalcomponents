#-*- coding: <utf-16> -*- 
import codecs
import os
import sys
import string
import operator
mywords=dict()
from math import sqrt
from  collections import defaultdict 
import numpy
from numpy import *
import time
#from pyx import *


GraphicsFlag = False

#-----------------------------------------------------------------------#
#									#
#	This program takes a trigram file and a word list		#	
#	and creates a file with lists of most similar words.		#
#	John Goldsmith and Wang Xiuli 2012.				#
#									#
#-----------------------------------------------------------------------#

#---------------------------------------------------------------------------#
#	Variables to be changed by user
#---------------------------------------------------------------------------#
LatexFlag = True
PrintEigenvectorsFlag = True
unicodeFlag = False 
FileEncoding =  "ascii"

datafilelocation 	=  "../../data/"
languagename 		= "english-browncorpus"
  
wordfolder    		= datafilelocation + languagename + "/dx1/"
trigramfolder 		= datafilelocation + languagename + "/ngrams/"
outfolder     		= datafilelocation + languagename + "/neighbors/"

outfile_eps 		= outfolder + "dx2_files/results/" + languagename + ".eps" 
outfile_pdf 		= outfolder + "dx2_files/results/" + languagename + ".pdf" 

NumberOfWordsForContext 	= 1000 # 40000
NumberOfEigenvectors 		= 11
NumberOfWordsForAnalysis	= 500 #4000
NumberOfNeighbors  		= 9

punctuation 		= " $/+.,;:?!()\"[]"


if NumberOfWordsForAnalysis > NumberOfWordsForContext:
	print "No -- the number of words to be shown must be no larger than the number of words used for the matrix."



#---------------------------------------------------------------------------#
#	File names
#---------------------------------------------------------------------------#

 
infileTrigramsname 	= trigramfolder + languagename + "_trigrams.txt"
infileWordsname 	= wordfolder 	+ languagename + ".dx1" 
outfilenameEigenvectors	= outfolder 	+ languagename + "_PoS_words_eigenvectors" + ".txt"
outfilenameNeighbors	= outfolder 	+ languagename + "_PoS_closest" + "_" + str(NumberOfNeighbors ) + "_neighbors.txt"
outfilenameLatex 	= outfolder 	+ languagename + "_latex.tex"
outfilenameContexts     = outfolder 	+ languagename + "_contexts.txt"
outfilenameEigenCoordinates = outfolder + languagename + "_eigen_coordinates.txt"

print "\n\nI am looking for: ", infileTrigramsname

#---------------------------------------------------------------------------#
#	Variables
#---------------------------------------------------------------------------#

linecount 		= 0
contexts 		= dict()

analyzedwordlist 	= list() # this means that the info comes from the independent word file
analyzedworddict   	= dict()


#---------------------------------------------------------------------------#
#	Open files for reading and writing
#---------------------------------------------------------------------------#

if unicodeFlag:
	trigramfile 		=codecs.open(infileTrigramsname, encoding = FileEncoding)
	wordfile 		=codecs.open(infileWordsname, encoding = FileEncoding)
	if PrintEigenvectorsFlag:
		outfileEigenvectors = codecs.open (outfilename1, "w",encoding = FileEncoding)
	outfileNeighbors	= codecs.open (outfileneighborsname, "w",encoding = FileEncoding)
	
else:
	if PrintEigenvectorsFlag:
		outfileEigenvectors = open (outfilenameEigenvectors, "w")
	outfileNeighbors	= open (outfilenameNeighbors, "w")
	outfileLatex 		= open (outfilenameLatex, "w")
	outfileContexts 	= open (outfilenameContexts, "w")
	eigencoordinates 	= open (outfilenameEigenCoordinates, "w")

	wordfile		= open(infileWordsname)
	trigramfile 		= open(infileTrigramsname)

print "Language is", languagename, ". File name:", languagename, ". Number of words", NumberOfWordsForContext, "."

if PrintEigenvectorsFlag:
	print >>outfileEigenvectors,"#", \
			languagename, "\n#", \
			languagename,"\n#", \
			"Number of words used for matrix", NumberOfWordsForContext, "\n#",\
			"Number of words analyzed", NumberOfWordsForAnalysis,"\n#", \
			"Number of neighbors identified", NumberOfNeighbors, "\n#","\n#" 

print >>outfileNeighbors, "#", \
		languagename, "\n#",\
		languagename, "\n#",\
		"Number of words used for context", NumberOfWordsForContext,"\n#", \
		"Number of words analyzed", NumberOfWordsForAnalysis,"\n#", \
		"Number of neighbors identified", NumberOfNeighbors,"\n#","\n#"

 
#---------------------------------------------------------------------------#
#	Read trigram file
#---------------------------------------------------------------------------#
from_word_to_context = dict()
context_list = list()
if True:
	for line in wordfile:
		pieces = line.split()
		if pieces[0] == "#":
			continue
		if len(pieces) < 3:
			continue
		mywords[pieces[0]] = int(pieces[1])		 
	print "1. Word file is ", infileWordsname		
	wordfile.close()
	analyzedwordlist = sorted(mywords,key=mywords.__getitem__,reverse=True)
	analyzedwordlist[NumberOfWordsForAnalysis:] = []
	for i in range(NumberOfWordsForAnalysis):
		analyzedworddict[analyzedwordlist[i]] = i
 	 
	print "2. Reading in trigram file."
	for line in trigramfile:
		linecount += 1
		line = line.split()
		if line[0] == "#":
			continue		 
		thesewords = line # line[0].split() #was split on "_" 
		#print line
		focus_word = thesewords[1]
		if focus_word in analyzedworddict:
			context = thesewords[0] + "_" +  thesewords[2]
			if not context in contexts:
				contexts[context] = dict()
				context_list.append(context)
			contexts[context][focus_word] = 1
			if not focus_word  in from_word_to_context:
				from_word_to_context[focus_word] = list()
			from_word_to_context[focus_word].append(context)

		#Left trigrams
		focus_word = thesewords[0]
		if focus_word in analyzedworddict:
			context = "_" + thesewords[1] + "+" + thesewords[2]
			if not context in contexts:
				contexts[context] = dict()
			contexts[context][focus_word] = 1
			if focus_word not in from_word_to_context:
				from_word_to_context[focus_word] = list()
			from_word_to_context[focus_word].append(context)


		#Right trigrams
		focus_word = thesewords[2]
		if focus_word in analyzedworddict:	
			context = thesewords[0]  + "+" + thesewords[1] + "_"
			if not context in contexts:
				contexts[context] = dict()
			contexts[context][focus_word] = 1
 			if focus_word not in from_word_to_context:
				from_word_to_context[focus_word] = list()
			from_word_to_context[focus_word].append(context)
 

#contextlist = contexts.keys() 
 
print "%-50s %3d = number of contexts" % ("3. End of words and counts.", len(context_list) ) 

#---------------------------------------------------------------------------#
#	Make zero-mean incidence graph
#---------------------------------------------------------------------------#
print  "%-50s" %  "3a. Make zero-mean incidence graph....",
incidencegraph= zeros( (NumberOfWordsForAnalysis,len(context_list)) ) 
for contextno in range(len(context_list)):
	this_sum = 0.0
	print contextno
	for wordno in range(NumberOfWordsForAnalysis):
		this_sum += len(from_word_to_context[analyzedwordlist[wordno] ])
#---------------------------------------------------------------------------#
#	Count context features shared by words
#---------------------------------------------------------------------------#

print "%-50s" % "4. Counting context features shared by words...",
NearbyWords = zeros( (NumberOfWordsForAnalysis,NumberOfWordsForAnalysis) )
count = 0 
for context in contexts:
	count += 1
#	if count%10000 == 0:
#		print count/10000, 
	if len(contexts[context]) == 1:
		continue
	for word1 in contexts[context]:	 
		w1 = analyzedworddict[word1]	
		for word2 in contexts[context]: 
			w2 = analyzedworddict[word2]
			if not w1 == w2:		
				NearbyWords[w1,w2] += 1
print "Done.", count
#---------------------------------------------------------------------------#
#	Normalize.
#---------------------------------------------------------------------------#
print  "%-50s" % "5. Normalizing nearness measurements....",
Diameter = defaultdict()
count = 0
for w1 in range(NumberOfWordsForAnalysis):
	for w2 in range(NumberOfWordsForAnalysis):
		if w1 == w2:
			continue
		if not w1 in Diameter:
			Diameter[w1] = 0		 
		Diameter[w1] += NearbyWords[w1,w2]
		count += 1
print "Done.", count
#---------------------------------------------------------------------------#
#	Incidence graph
#---------------------------------------------------------------------------#
print  "%-50s" %  "6. We compute the incidence graph....",
incidencegraph= zeros( (NumberOfWordsForAnalysis,NumberOfWordsForAnalysis) )
count = 0
for w1 in range( NumberOfWordsForAnalysis ):
	for w2 in range( NumberOfWordsForAnalysis ):
		if w1 == w2:
			incidencegraph[w1,w1] = Diameter[w1]
		else:
			incidencegraph[w1, w2] = NearbyWords[w1,w2]	
			count += 1

print "Done.", count 
 		 
#---------------------------------------------------------------------------#
#	Zero mean
#---------------------------------------------------------------------------#



#---------------------------------------------------------------------------#
#	Normalize the laplacian.
#---------------------------------------------------------------------------#
print  "%-50s" %  "7. We normalize the laplacian....",
#Normalize the laplacian:
count = 0
mylaplacian = zeros((NumberOfWordsForAnalysis,NumberOfWordsForAnalysis) )
for i in range(NumberOfWordsForAnalysis):
	mylaplacian[i,i] = 1
	for j in range(NumberOfWordsForAnalysis):
		if not i == j:
			if incidencegraph[i,j] == 0:
				mylaplacian[i,j]=0
			else:
				mylaplacian[i,j] = -1 * incidencegraph[i,j]/ math.sqrt ( Diameter[i] * Diameter[j] )
				count += 1		 	
print "Done.", count 
 
#---------------------------------------------------------------------------#
#	Compute eigenvectors.
#---------------------------------------------------------------------------#
print "%-50s" %  "8. Compute eigenvectors...",
myeigenvalues, myeigenvectors = numpy.linalg.eigh(mylaplacian)
print "Done."

formatstr = '%15d  %15s %10.3f'
 
  
#---------------------------------------------------------------------------#
#	Generate latex output.
#---------------------------------------------------------------------------#
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

for wordno in range(NumberOfWordsForAnalysis):
	#print                     analyzedwordlist[wordno], myeigenvectors[wordno,0],myeigenvectors[wordno,1],myeigenvectors[wordno,2]
	print >>eigencoordinates, analyzedwordlist[wordno], myeigenvectors[wordno,0],myeigenvectors[wordno,1],myeigenvectors[wordno,2]
	coordinates.append((myeigenvectors[wordno,1],myeigenvectors[wordno,2]) )
	
eigencoordinates.close()

#PUT THESE BACK IN
#g.plot(graph.data.points(coordinates,x=1,y=2))
#g.writeEPSfile(outfile_eps)
#g.writePDFfile(outfile_pdf)
#outfile_eps.close()
#outfile_pdf.close()

print "Printing to ", outfolder 
	 
if PrintEigenvectorsFlag:

	for eigenno in range(NumberOfEigenvectors):
		print >>outfileEigenvectors
		print >>outfileEigenvectors,headerformatstr %("Eigenvector number", "word" , myeigenvalues[eigenno])
		print >>outfileEigenvectors,"_____________________________________________" 
		for wordno in range(NumberOfWordsForAnalysis):
			print >>outfileEigenvectors, formatstr %(eigenno, analyzedwordlist[wordno], myeigenvectors[wordno,eigenno])

if LatexFlag:
	for eigenno in range(NumberOfEigenvectors):
		eigenlist=list()	
		#data[eigenno] = list()
		data = list()
		for wordno in range (NumberOfWordsForAnalysis):		 
			eigenlist.append( (wordno,myeigenvectors[wordno, eigenno]) )			
		eigenlist.sort(key=lambda x:x[1])			
		print >>outfileLatex			 
		print >>outfileLatex, "Eigenvector number", eigenno, "\n" 
		print >>outfileLatex, "\\begin{tabular}{lll}\\toprule"
		print >>outfileLatex, " & word & coordinate \\\\ \\midrule "

		for i in range(NumberOfWordsForAnalysis):			 
			word = analyzedwordlist[eigenlist[i][0]]
			coord =  eigenlist[i][1]
			if i < NumberOfWordsToDisplayForEachEigenvector or i > NumberOfWordsForAnalysis - NumberOfWordsToDisplayForEachEigenvector:
				data.append((i, word , coord ))
		
				#for eigenno in data.keys():
				
		for (i, word, coord) in data:
			if word == "&":
				word = "\&" 
			print >>outfileLatex,  "%5d & %10s &  %10.3f \\\\" % (i, word, coord) 

		print >>outfileLatex, "\\bottomrule \n \\end{tabular}", "\n\n"
		print >>outfileLatex, "\\newpage" 
print >>outfileLatex, "\\end{document}" 
#---------------------------------------------------------------------------#
#	Finding coordinates in space of low dimensionality
#---------------------------------------------------------------------------#
print "10. Finding coordinates in space of low dimensionality."
 
coordinates 		= dict()
wordsdistance 		= dict()
closestNeighbors 	= dict() #a dict whose values are lists; the lists are the closest words to the key.
 
thislist = list() 
 
for wordno in range(NumberOfWordsForAnalysis):
	coordinates[wordno]= list() 
	for eigenno in range (1,NumberOfEigenvectors):
		coordinates[wordno].append ( myeigenvectors[ wordno, eigenno ] )

for wordno1 in range(NumberOfWordsForAnalysis):	 

	word = analyzedwordlist[wordno1]
 
       	wordsdistance[word] = list()

	for wordno2 in range (NumberOfWordsForAnalysis):		 
		distance = 0
		for coordno in range(NumberOfEigenvectors-1):
			x = coordinates[wordno1][coordno] - coordinates[wordno2][coordno]
			distance += abs(x * x * x)		 
                wordsdistance[word].append((wordno2,distance))		 
 



#---------------------------------------------------------------------------#
#	 Finding closest neighbors on the manifold's approximation
#---------------------------------------------------------------------------#

print "11. Finding closest neighbors on the manifold('s approximation)."
		 
for wordno1 in range(NumberOfWordsForAnalysis): 
    
	word1 = analyzedwordlist[wordno1]		
	if not word1 in closestNeighbors:
		closestNeighbors[word1] = list()
	wordsdistance[word1].sort(key=lambda x:x[1])     
	print >>outfileNeighbors, word1,
	count = 0
 

 	for (wordno2, distance) in wordsdistance[word1]:		 
		if wordno1 == wordno2:			 
			continue			
		count += 1
		word2 = analyzedwordlist[wordno2]	 
		print >>outfileNeighbors, word2, 
		closestNeighbors[word1].append(word2)	 	 
		if count >= NumberOfNeighbors:
			break
	print >>outfileNeighbors

outfileNeighbors.close()

#---------------------------------------------------------------------------#
#	 Print contexts shared by nearby words
#---------------------------------------------------------------------------#
numberperrow= 5
for word in analyzedwordlist:
	print >>outfileContexts,"\n", word,"\n\t",
	number = 1
	if (False):
		for context in from_word_to_context[word]:
			if len(contexts[context]) >100:
				print >>outfileContexts, "%-25s %3d " % ( context, len(contexts[context])),
				number += 1
				if number == numberperrow:
					number = 1
					print >>outfileContexts, "\n\t",
	these_contexts = set( from_word_to_context[word] )
	for word2 in closestNeighbors[word]:
		print >>outfileContexts, word2
		these_contexts.intersection( set(from_word_to_context[word2]))
	for context in these_contexts: 	
		if len(contexts[context]) >100:
			print >>outfileContexts, "%-25s %3d " % ( context, len(contexts[context])),
			number += 1
			if number == numberperrow:
				number = 1
				print >>outfileContexts, "\n\t",
 
print "Exiting successfully."

#os.popen("latex " + outfilenameLatex ) 

if PrintEigenvectorsFlag:
	outfileEigenvectors.close()
outfileNeighbors.close()
 



