import numpy as np
import random

def should_mutate(prob):
    if(np.random.uniform(0, 100) <= prob):
        return True
    return False

def mutate(f, fj, prob, mutations):
    '''
    mutate f and its corresponding conjugates at a desired number of points

    parameters:
        f: numpy array of fourier transform of some sound sample
        fj: complex conjugate of f
        prob: independent probability that a mutation will occur
        mutations: number of desired mutations to try to induce

    return:
        mutated f and fj
    '''
    mutation_points = random.sample(range(0, len(f)), mutations)
    for i in range(len(mutation_points)):
        if(should_mutate(prob)):
            orig = f[mutation_points[i]]
            new_real = np.random.uniform(0, 10 * orig.real)
            new_imag = np.random.uniform(0, 10 * orig.imag)
            f[mutation_points[i]] = complex(new_real, new_imag)
            fj[mutation_points[i]] = np.conj(complex(new_real, new_imag))