import math
import numpy as np
### SHIMA ALGORITHM

# Let I = [0,1,2,3,...,N-1] be the list of all particle indices
# make a random permutation of I by
# current number of particles in cell
no_spct = 20
# permutation = np.ones(len(masses),dtype=int)

# returns a permutation of the list [0,1,2,..,N-1]
def generate_permutation(N):
    permutation = np.zeros(N, dtype=int)
    for n_next in range(1, N):
        q = np.random.randint(0,n_next+1)
        if q==n_next:
            permutation[n_next] = n_next
        else:
            permutation[n_next] = permutation[q]
            permutation[q] = n_next
    return permutation

permutations = []
# start from a permutation of length n: a_n = [i0, i1, i2, i3, ... i(n-1)]
# choose q randomly from {0,1,2,3,...,n}
print("number of indices in the list [0,1,2,...]:", no_spct)
print(f"there are {math.factorial(no_spct):.2e} , possible permutations")
no_perm = 100000
print("building", no_perm, "permutations with Shimas method")
strange_things = False
for i in range(no_perm):
    permutations.append(generate_permutation(no_spct))

for i,perm in enumerate(permutations):
    unique, cnts = np.unique(perm, return_counts=True)
    unique2 = np.unique(cnts, return_counts=True)
    if (np.amax(perm) != no_spct-1 or np.amin(perm) != 0
        or len(unique2[0]) != 1
        or unique2[0][0] != 1
        or unique2[1][0] != no_spct):
        strange_things = True
        print(i, np.amax(perm), np.amin(perm), unique2)
    # if np.amax(cnts>1): print(i, np.amax(cnts>1))

for i,perm in enumerate(permutations):
    for j in range(i+1,no_spct):
        perm2 = permutations[j]
        isequal = np.array_equal(perm,perm2) 
        if isequal:
            print(i, j, isequal)

print()
if not strange_things:
    print("no unexpected behavior for the permutations:")
    print("every permutation includes all", no_spct, "indices exactly once")
    print("no permutation occurs more than once (every one is unique)")