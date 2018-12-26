class gradDescend():
    def __init__(self, function, x_0 = None, ):
        self.function = function

        self.x_0 = x_0 == None x_0 else 
    
    def GSD(self):
        x = self.x_0
        #controlli vari sulla funzione
        y, g = self.function.calculate(x)
        if norm(g) == 0:
            #ho l'ottimo
        while norm g != 0 or numIter <= soglia:
            alpha = self.function.exactSearch(x, g)
            x = x - alpha*g
            y, g = self.function.calculate(x)
