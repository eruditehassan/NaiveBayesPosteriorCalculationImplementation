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

The instantition wil be done as follows: `customNB = NaiveBayesPosterior(df, features, target, lam)`.

### Prior Calculation
The prior can be calculated by using `customNB.calculate_prior()` where *customNB* is the name of the instantiated object. The value of prior can be accessed using `customNB.prior`.

### Likelihood Calculation
The prior can be calculated by using `customNB.calculate_likelihood()` where *customNB* is the name of the instantiated object. The value of prior can be accessed using `customNB.likelihood`.

### Posterior Calculation
There are two ways to calculate posterior.

**Method 1** 

First calculate `prior` and `likelihood` using functions mentioned above and once that is done you can use `customNB.calculate_posterior(values)` where *customNB* is the name of the instantiated object, and **values** are the values for which posterior is to calculated.

The posterior calculation function takes input of a nested list of the following format
`values = [['weather','Sunny'],['temperature','Hot'], 'Yes']`

where `weather` is the column name and `Sunny` is the value.


**Method 2** 

Directly calculate the posterior by using `customNB.complete_calculation(values)` where *customNB* is the name of the instantiated object, and **values** are the values for which posterior is to calculated.

The posterior calculation function takes input of a nested list of the following format
`values = [['weather','Sunny'],['temperature','Hot'], 'Yes']`

where `weather` is the column name and `Sunny` is the value.
