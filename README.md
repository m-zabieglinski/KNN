# KNN
kohonen.py is a Python file that includes tools to create your own Kohonen Neural Network and train it. The networks are not accurate and rather slow to learn. This needs vast improvements, but the algorithms work.

The main feature is the KN class.

**KN** is a class, instances of which are Kohonen Neural Networks. It has methods:

  -create - which creates KNN of size given in when initializing the instance and with random weights
  
  -train - which trains the network on the given dataset
  
  -categorize - which categorizes the given dataset (maps it using the KN instance)
  
  -net - which simply returns the KNN (a 1D or 2D NumPy array of weights)
  
  -show - which is simply print(net)
  
  -showmap - which returns a plot of the network and an optional given dataset, works only for network spanned over 2D spaces
  
  
 **disk.py** and **square.py** are example implementations of KNN using kohonen.py on datasets representing a disk and a square, respectively. They create a KNN, train it, and then plot it over the dataset, using the different methods of KN.
