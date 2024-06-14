from scipy.fftpack import fft
from scipy.fftpack import ifft
import numpy as np
import random
from .mutation import mutate

def multi_recombine(population, data1, data2, prob, mutations, cross):
    '''
    recombine 2 distinct sound samples through random crossover points
    and mutate

    parameters:
        data1: 1D numpy array of first sound sample
        data2: 1D numpy array of second sound sample
        population: desired population size to be created
        cross: desired number of crossover points
        mutate: percent chance of each of the desired mutations occurs
        mutations: number of mutations to perform
        cross: number of crossover points

    return:
        a list of size population containing 1D numpy arrays
        of the recombined sound samples
    '''
    #resize the smaller sound data to match the larger sound data
    if(len(data1) != len(data2)):
        if(len(data1) < len(data2)):
            smaller = data1
            larger = data2
        else:
            smaller = data2
            larger = data1
        multiple = int(len(larger) / len(smaller))
        init = smaller
        if(multiple >= 2):
            for i in range(multiple - 1):
                init = np.concatenate((init, smaller))
        diff = len(larger) - multiple * len(smaller)
        init = np.concatenate((init, smaller[0:diff]))
        data1 = init
        data2 = larger
    dft1 = fft(data1)
    dft2 = fft(data2)
    if(len(dft1) % 2 == 1):
        chop = int(len(dft1) / 2) + 1
        next_chop = chop
    else:
        chop = int(len(dft1) / 2)
        next_chop = chop + 1
    #focusing on keeping the left portion of f1
    f1 = dft1[1:chop]
    f1_conj = np.flip(dft1[next_chop:len(dft1)])
    f2 = dft2[1:chop]
    f2_conj = np.flip(dft2[next_chop:len(dft2)])
    resulting_list = []
    for i in range(int(population / 2)):
        max_range = range(int(len(f1) / 50), int(3 * len(f1) / 4))
        crossover_points = random.sample(max_range, cross)
        crossover_points.sort()

        # the first slice in the sequence is always from f1, while
        # the last slice is from f2 if the cross is odd otherwise it's f1
        result1_seq = [f1[0:crossover_points[0]]]
        result1_conj_seq = [f1_conj[0:crossover_points[0]]]
        for i in range(cross - 1):
            # if i even add from f2
            if(i % 2 == 0):
                result1_seq.append(f2[crossover_points[i]:crossover_points[i + 1]])
                result1_conj_seq.append(f2_conj[crossover_points[i]:crossover_points[i + 1]])
            else:
                result1_seq.append(f1[crossover_points[i]:crossover_points[i + 1]])
                result1_conj_seq.append(f1_conj[crossover_points[i]:crossover_points[i + 1]])
        if(cross % 2 == 1):
            result1_seq.append(f2[crossover_points[cross - 1]:len(f1)])
            result1_conj_seq.append(f2_conj[crossover_points[cross - 1]:len(f1)])
        else:
            result1_seq.append(f1[crossover_points[cross - 1]:len(f1)])
            result1_conj_seq.append(f1_conj[crossover_points[cross - 1]:len(f1)])

        result2_seq = [f2[0:crossover_points[0]]]
        result2_conj_seq = [f2_conj[0:crossover_points[0]]]
        for i in range(cross - 1):
            # if i even add from f1
            if(i % 2 == 0):
                result2_seq.append(f1[crossover_points[i]:crossover_points[i + 1]])
                result2_conj_seq.append(f1_conj[crossover_points[i]:crossover_points[i + 1]])
            else:
                result2_seq.append(f2[crossover_points[i]:crossover_points[i + 1]])
                result2_conj_seq.append(f2_conj[crossover_points[i]:crossover_points[i + 1]])
        if(cross % 2 == 1):
            result2_seq.append(f1[crossover_points[cross - 1]:len(f1)])
            result2_conj_seq.append(f1_conj[crossover_points[cross - 1]:len(f1)])
        else:
            result2_seq.append(f2[crossover_points[cross - 1]:len(f1)])
            result2_conj_seq.append(f2_conj[crossover_points[cross - 1]:len(f1)])

        result1 = np.concatenate(result1_seq)
        result1_conj = np.concatenate(result1_conj_seq)
        result2 = np.concatenate(result2_seq)
        result2_conj = np.concatenate(result2_conj_seq)
        mutate(result1, result1_conj, prob, mutations)
        mutate(result2, result2_conj, prob, mutations)
        assert(len(result1) == len(result1_conj))
        assert(len(result2) == len(result2_conj))
        assert(len(result1) == len(result2))
        assert(len(result1) == len(f1))

        if(len(dft1) % 2 == 1):
            beg = np.array([dft1[0]])
            final1 = np.concatenate((beg, result1, np.flip(result1_conj)))
            conv1 = np.real(ifft(final1))
            loudest1 = np.amax(np.absolute(conv1))
            resulting_list.append(np.float32(conv1 / loudest1))
            final2 = np.concatenate((beg, result2, np.flip(result2_conj)))
            conv2 = np.real(ifft(final2))
            loudest2 = np.amax(np.absolute(conv2))
            resulting_list.append(np.float32(conv2 / loudest2))
        else:
            beg = np.array([dft1[0]])
            mid = np.array([dft1[chop]])
            final1 = np.concatenate((beg, result1, mid, np.flip(result1_conj)))
            conv1 = np.real(ifft(final1))
            loudest1 = np.amax(np.absolute(conv1))
            resulting_list.append(np.float32(conv1 / loudest1))
            final2 = np.concatenate((beg, result2, mid, np.flip(result2_conj)))
            conv2 = np.real(ifft(final2))
            loudest2 = np.amax(np.absolute(conv2))
            resulting_list.append(np.float32(conv2 / loudest2))
    return resulting_list

def single_recombine(population, data1, data2, prob, mutations):
    '''
    recombine 2 distinct sound samples through 1 random crossover point
    and mutate

    parameters:
        sampling_freq: Sampling frequency of both sound samples
        data1: 1D numpy array of first sound sample
        data2: 1D numpy array of second sound sample
        population: desired population size to be created
        cross: desired number of crossover points
        mutate: percent chance of each of the desired mutations occurs
        mutations: number of mutations to perform

    return:
        a list of size population containing 1D numpy arrays
        of the recombined sound samples
    '''
    #resize the smaller sound data to match the larger sound data
    if(len(data1) != len(data2)):
        if(len(data1) < len(data2)):
            smaller = data1
            larger = data2
        else:
            smaller = data2
            larger = data1
        multiple = int(len(larger) / len(smaller))
        init = smaller
        if(multiple >= 2):
            for i in range(multiple - 1):
                init = np.concatenate((init, smaller))
        diff = len(larger) - multiple * len(smaller)
        init = np.concatenate((init, smaller[0:diff]))
        data1 = init
        data2 = larger
    dft1 = fft(data1)
    dft2 = fft(data2)
    if(len(dft1) % 2 == 1):
        chop = int(len(dft1) / 2) + 1
        next_chop = chop
    else:
        chop = int(len(dft1) / 2)
        next_chop = chop + 1
    #focusing on keeping the left portion of f1
    f1 = dft1[1:chop]
    f1_conj = np.flip(dft1[next_chop:len(dft1)])
    f2 = dft2[1:chop]
    f2_conj = np.flip(dft2[next_chop:len(dft2)])
    resulting_list = []
    for i in range(int(population / 2)):
        crossover_point = random.randint(int(len(f1) / 50), int(3 * len(f1) / 4))
        result1 = np.concatenate((f1[0:crossover_point], f2[crossover_point:len(f2)]))
        result1_conj = np.concatenate((f1_conj[0:crossover_point],
            f2_conj[crossover_point:len(f2)]))
        result2 = np.concatenate((f2[0:crossover_point], f1[crossover_point:len(f1)]))
        result2_conj = np.concatenate((f2_conj[0:crossover_point],
            f1_conj[crossover_point:len(f1)]))
        mutate(result1, result1_conj, prob, mutations)
        mutate(result2, result2_conj, prob, mutations)
        if(len(dft1) % 2 == 1):
            beg = np.array([dft1[0]])
            final1 = np.concatenate((beg, result1, np.flip(result1_conj)))
            conv1 = np.real(ifft(final1))
            loudest1 = np.amax(np.absolute(conv1))
            resulting_list.append(np.float32(conv1 / loudest1))
            final2 = np.concatenate((beg, result2, np.flip(result2_conj)))
            conv2 = np.real(ifft(final2))
            loudest2 = np.amax(np.absolute(conv2))
            resulting_list.append(np.float32(conv2 / loudest2))
        else:
            beg = np.array([dft1[0]])
            mid = np.array([dft1[chop]])
            final1 = np.concatenate((beg, result1, mid, np.flip(result1_conj)))
            conv1 = np.real(ifft(final1))
            loudest1 = np.amax(np.absolute(conv1))
            resulting_list.append(np.float32(conv1 / loudest1))
            final2 = np.concatenate((beg, result2, mid, np.flip(result2_conj)))
            conv2 = np.real(ifft(final2))
            loudest2 = np.amax(np.absolute(conv2))
            resulting_list.append(np.float32(conv2 / loudest2))
    return resulting_list