# Naive Bayes Posterior Calculation Implementation
Custom Class based implementation of Naive Bayes Posterior Calculation in Python


## Documentation
The details of what each function does and how to use it is given below.

### Object Instatiation
The class can be intantitiated by simply providing following parameters:

1. `df` - Pandas Data Frame containing the data on which calculations are to be done.
2. `features` - A list containing the names of all the columns of data frame which are features.
3. `target` - The name of target column of the data frame.
4. `lam` - The Laplace Smoothing parameter, by default it is 0.

