import methods as methods
import numpy as np

#open the text file and read the values
with open('Data\instances.txt', 'r') as f:
        input_data = [list(map(int, line.split(','))) for line in f if line!='\n']
        


#A,b,c = methods.importLP('simplicity.lp')


#main
A,b,c = methods.getInstance(input_data,0)

kMax = 1
isMaximization = True

if(isMaximization):
   c = -1 * c

#c = methods.repeat_elements(c, kMax)

#lambda, the penalty multiplier, is p
p=0.01



Qdict = methods.Qdict(A=A, b=b, c=c, kMax = kMax, p=p)

#Qt = np.array([[2, -4], [-4, 3]])
#Qtdict = dict(np.ndenumerate(-1 * Qt))



import neal
sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(Qdict, chain_strength = 10, num_reads = 5)
print(sampleset.first.energy)
print(sampleset)


