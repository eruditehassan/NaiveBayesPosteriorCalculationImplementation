class NaiveBayesPosterior():
    def __init__(self, df, features, target, lam = 0):
        self.features = features
        self.target = target
        self.df = df
        self.prior = 0
        self.likelihood = {}
        self.posterior = 0
        self.lam = lam
        
    def calculate_prior(self):
        if (self.lam == 0):
            self.prior = self.df.groupby(self.target).size().div(len(self.df))
        else:
            size = self.df.groupby(self.target).size() + self.lam
            total = self.df.groupby(self.target).size().sum() + \
                self.lam * len(self.df.groupby(self.target).size().index)
            self.prior = size / total
        
    def calculate_likelihood(self):
        if (self.lam == 0):    
            for feature in self.features:
                self.likelihood[feature] = self.df.groupby([self.target,feature]).size().div(len(self.df)).div(self.prior)
        else:
            for feature in self.features:
                size = self.df.groupby([self.target,feature]).size() + self.lam
                count = len(self.df) + self.lam*(len(self.df.groupby(feature).size().index))
                self.likelihood[feature] = (size / count) / self.prior
        
            
    def calculate_posterior(self, values):
        self.post_calc = {}
        self.post_calc['Yes'] = self.likelihood[values[0][0]]['Yes'][values[0][1]] * \
            self.likelihood[values[1][0]]['Yes'][values[1][1]] * self.prior['Yes']
        
        self.post_calc['No'] = self.likelihood[values[0][0]]['No'][values[0][1]] * \
            self.likelihood[values[1][0]]['No'][values[1][1]] * self.prior['No']
        
        self.denominator_factor = self.post_calc['Yes'] + self.post_calc['No']
        
        self.posterior = self.post_calc[values[-1]] / self.denominator_factor
        
        return self.posterior
    
    def complete_calculation(self, values):
        self.calculate_prior()
        self.calculate_likelihood()
        self.calculate_posterior(values)