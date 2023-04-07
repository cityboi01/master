import methods as methods

#open the Knapsack test instances file and read the data
with open('Data\instances.txt', 'r') as f:
        input_data = [list(map(int, line.split(','))) for line in f if line!='\n']
        

# gets matrix components matrices A,b,c from a .lp CPLEX file
#A,b,c = methods.importLP('simplicity.lp')


#gets matrix component matrices A,b,c for a Knapsack instance
A,b,c = methods.getInstance(input_data,0)

#number of qubits per variable
kMax = 1

isMaximization = True
if(isMaximization):
   c = -1 * c


#lambda, the penalty multiplier, is p
p=0.01



Qdict = methods.Qdict(A=A, b=b, c=c, kMax = kMax, p=p)

#call simulated annealing sampler on Qdict
import neal
sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(Qdict, chain_strength = 10, num_reads = 5)
print(sampleset.first.energy)
print(sampleset)


