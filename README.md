# KNN
kohonen.py is a Python file that includes tools to create your own Kohonen Neural Network and train it. The networks are not accurate and rather slow to learn. This needs vast improvements, but the algorithms work.

First, 2 functions are defined inside it:

  **distance(v1, v2)** which gives the euclidean distance between vectors v1 and v2 (python lists)
  
  **normalize(dataset)** which normalizes the given dataset (changes values from range [A,B] to [0,1])
  
  
  These 2 functions are later used by the KN class.
  

**KN** is a class instances of which are Kohonen Neural Networks. It has methods:

  -create - which creates KNN of size given in when initializing the instance and with random weights
  
  -train - which trains the network on the given dataset
  
  -categorize - which categorizes the given dataset (maps it using the KN instance)
  
  -net - which simply returns the KNN (a 1D or 2D NumPy array of weights)
  
  -show - which is simply print(net)
  
  -showmap - which returns a plot of the network and an optional given dataset, works only for network spanned over 2D spaces
