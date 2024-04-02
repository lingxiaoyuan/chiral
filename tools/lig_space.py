import math

def sumf(c,f,x):
    s=0
    for i in range(len(c)):
        s+=c[i]*f(i,x)
    return s

#poly_space = mix_space(name = ['poly']*10)
#sin_space = mix_space(name = ['sin']*10)

class mix_space():
    def __init__(self,name = ['poly']*10):
        self.name = name
        
    def f(self,i,x):
        if self.name[i] == 'sin':
            #return math.sin(i*x*math.pi/self.L/2)
            return math.sin(i*x)
        elif self.name[i] == 'sint':
            return math.sin(x+i)
        elif self.name[i] == 'poly':
            return x**i
    

    def f_diff(self,i,x):
        if self.name[i] == 'sin':
            return i*math.cos(i*x)
        elif self.name[i] == 'sint':
            return math.cos(x+i)
        elif self.name[i] == 'poly':
            return i*x**(i-1)

    def f_diff2(self,i,x):
        if self.name[i] == 'sin':
            return -i*i*math.sin(i*x)
        elif self.name[i] == 'sint':
            return -math.sin(x+i)
        elif self.name[i] == 'poly':
            return i*(i-1)*x**(i-2)