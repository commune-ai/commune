from .recombination import multi_recombine
from .objective_function import fitness
import numpy as np
from scipy.io import wavfile

def evolve(init_pop, iterations, preferred_chord, preferred_scale, output_dir, desired_prob, desired_mutations, desired_crossover_points):
    '''
    Use genetic algorithms process to evolve an initial population of
    sound samples

    parameters:
        init_pop: initial population of candidate sound samples; population
            size at each iteration will also be of size init_pop
        iterations: number of iterations to evolve the sound samples

    return:
        population after several evolution rounds
    '''
    pop = init_pop
    p1 = None
    p2 = None
    for i in range(iterations):
        fitnesses = np.zeros(len(pop))
        for j in range(len(fitnesses)):
            fitnesses[j] = fitness(pop[j], preferred_chord, preferred_scale)
        ordered_fitness = (np.argsort(fitnesses))[::-1]
        p1 = pop[ordered_fitness[0]]
        p2 = pop[ordered_fitness[1]]
        if(i == 0):
            wavfile.write(output_dir + '/parent1.wav', 44100, p1)
            wavfile.write(output_dir + '/parent2.wav', 44100, p2)
        pop = multi_recombine(len(pop), p1, p2, desired_prob,
            desired_mutations, desired_crossover_points)
    return pop