# -*- coding: utf-8 -*-
"""
Created on Sun Mar 03 11:58:16 2019

@author: ercan
"""
import numpy as np
import math
import statistics
###
### This part of the code is for generating the feature_matrix for the students to process
###
file = open("datatraining.txt","r")
count = 0

#initialize feature_matrix
feature_matrix = np.zeros((1,6))

#Repeat for each instance in the text file
for line in file:
  
  #split the line into an array called "fields" using the "," as a separator:
  fields = line.split(",")
  
  # skip the first line
  if count == 0:
      count = count + 1
      continue 
  
  # we are not interested in the first two colums, so they are not included
  feature_matrix = np.concatenate((feature_matrix,[[float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7])]]))
  count = count + 1
  
feature_matrix = np.delete(feature_matrix, 0, 0) #remove the first row from the feature_matrix
file.close()
###
### end of feature_matrix generation
###


def calc_mean(feat):
    
    mean = sum(feat)/len(feat)
    print(mean)
    
    return mean

    
def calc_std(feat):
    
    n = len(feat)
    mean = sum(feat) /len(feat)
    var = sum([(x - mean)**2 for x in feat]) / n
    std = math.sqrt(var)
    
    return std

def calc_corr_coef(feat_a, feat_b):
    
    n = len(feat_a)
    meanf1 = calc_mean(feat_a)
    meanf2 = calc_mean(feat_b)
    
    a = 0
    fa1 = 0
    fb1 = 0
    for i in range(n):
        fa = feat_a[i] - meanf1
        fb = feat_b[i] - meanf2
        a += fa * fb
        fa1 += fa * fa
        fb1 += fb * fb
        coef = a / math.sqrt (fa1*fb1)
    return coef

# create feature vectors
feat_1 = feature_matrix[:,0]
feat_2 = feature_matrix[:,1] 
feat_3 = feature_matrix[:,2]
feat_4 = feature_matrix[:,3]
feat_5 = feature_matrix[:,4]
class_labels = feature_matrix[:,5]

feat_1_mean = calc_mean(feat_1)
feat_2_mean = calc_mean(feat_2)
feat_3_mean = calc_mean(feat_3)
feat_4_mean = calc_mean(feat_4)
feat_5_mean = calc_mean(feat_5)

feat_1_std = calc_std(feat_1)
feat_2_std = calc_std(feat_2)
feat_3_std = calc_std(feat_3)
feat_4_std = calc_std(feat_4)
feat_5_std = calc_std(feat_5)

corr_coef_1 = calc_corr_coef(feat_1, feat_2)
corr_coef_2 = calc_corr_coef(feat_2, feat_3)
corr_coef_3 = calc_corr_coef(feat_3, feat_4)
corr_coef_4 = calc_corr_coef(feat_4, feat_5)
corr_coef_5 = calc_corr_coef(feat_5, feat_1)

print ("Mean of the feature vectors")
print ("feat_1 mean is:", feat_1_mean) 
print ("feat_2 mean is:", feat_2_mean)
print ("feat_3 mean is:", feat_3_mean)
print ("feat_4 mean is:", feat_4_mean)
print ("feat_5 mean is:", feat_5_mean)
print ("***")

print ("Standard Deviation of the feature vectors")
print ("feat_1 std is:", feat_1_std)
print ("feat_2 std is:", feat_2_std)
print ("feat_3 std is:", feat_3_std)
print ("feat_4 std is:", feat_4_std)
print ("feat_5 std is:", feat_5_std)
print ("***")

print ("Some correlation coefficients of the feature vectors")
print ("corrrelation coef 1 is:", corr_coef_1)
print ("corrrelation coef 2 is:", corr_coef_2)
print ("corrrelation coef 3 is:", corr_coef_3)
print ("corrrelation coef 4 is:", corr_coef_4)
print ("corrrelation coef 5 is:", corr_coef_5)
print ("***")


"""
here you should add verification codes
in this part you need to make the same calculcations: mean, standard deviation and correlation coefficient
but this time you will use the built-in pyton functions
then you will compare the results with the ones you obtained using your own functions
these results should be printed on screen as well
"""

print ("feat_1 mean is:", np.mean(feat_1)) 
print ("feat_2 mean is:", np.mean(feat_2))
print ("feat_3 mean is:", np.mean(feat_3))
print ("feat_4 mean is:", np.mean(feat_4))
print ("feat_5 mean is:", np.mean(feat_5))
print ("***")

print ("feat_1 std is:", np.std(feat_1)) 
print ("feat_2 std is:", np.std(feat_2))
print ("feat_3 std is:", np.std(feat_3))
print ("feat_4 std is:", np.std(feat_4))
print ("feat_5 std is:", np.std(feat_5))
print ("***")

print ("feat_1 corrcoef is:", np.corrcoef(feat_1,feat_2)) 
print ("feat_2 corrcoef is:", np.corrcoef(feat_2,feat_3))
print ("feat_3 corrcoef is:", np.corrcoef(feat_3,feat_4))
print ("feat_4 corrcoef is:", np.corrcoef(feat_4,feat_5))
print ("feat_5 corrcoef is:", np.corrcoef(feat_5,feat_1))
print ("***")







