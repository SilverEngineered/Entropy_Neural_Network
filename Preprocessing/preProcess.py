import os
import csv
import numpy as np
def readDataFromCSV(path,delimiter=";"):
    """ Reads in data from CSV and returns it as lists
      
      Assumes the first line of the file indicates specific
      features.  Assumes the last element of each line is the label.

      Inputs: path - string of the path and name of the csv to read data from
              delimiter - character that seperates values in csv              

      Outputs: features - list of data points and features of length 
                          equal to the number of data points, each
                          element is of size (len features)
               labels - list of all labels [int]
    """
    labels = []
    features = []
    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        count = 0
        for row in reader:
            if(count!=0):
                features.append(row[:-1])
                labels.append(row[-1])
            count+=1
    return features,labels

def writeToCSV(data,path):
    """ Writes data as a csv to a specified path

      Inputs: data - list containing all of the data to write
              path - string of the path of the csv to write to
              (Clears previous file with the same name if it already exists)
      Outputs:  N/A
    """
    with open(path, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for row in data:
            writer.writerow(row)

def splitData(data):
    """ Splits data to test, train, and validation
      Saves first 70% of data to train, next 20% to test, last 10% to validation
 
      Inputs: data - list containing all of the data to split
      
      Outputs: train - first 70% of data as a list
               test - next 20% of data as a list
               validation - last 10% of data as a list
    """
    count = len(data)
    train = data[:int(count*.7)]
    test = data[int(count*.7):int(count*.9)]
    validation = data[int(count*.9):]
    return train, test, validation
def scale_images(images,scaling=[0,1],dtype=np.float32):
    """  Scale an array of images to the specified scaling range
         inputs:
             images -> numpy array of images in the form (x,i,j)
             where x is the number of images
                   i is the number of rows
                   j is the number of columns
             scaling ->  integer tuple (min,max)
             where min is the minimum value after scaling
                   max is the maximum value after scaling
    
         outputs:
             numpy array of images scaled from min to max
    """
    min_data, max_data = [float(np.min(images)), float(np.max(images))]
    min_scale, max_scale = [float(scaling[0]), float(scaling[1])]
    data = ((max_scale - min_scale) * (images - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)

