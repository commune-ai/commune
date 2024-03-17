import commune as c
from scipy.io import wavfile
import os
import numpy as np
import random
from .src.genetic_algorithm import evolve
from .src.objective_function import names, SCALE_NAMES
import requests
import gradio as gr

file_list = ['Noize_Filter.wav', 'Chorus101.wav', 'Chunky.wav', 'ShoppingAB.wav', 'Joiner.wav', 'Mo_Chop.wav', 'L-R.wav', 'Eko_Creak.wav', 'Shorty.wav', 'MachineMan.wav', 'Moodie.wav', 'Tale_of_2.wav', 'Tok.wav', 'Chopper.wav', 'bIZZER.wav', 'ShoppingB.wav', 'Boingg.wav', 'Farmer.wav', 'sUBTLE_aRP.wav', 'FilteredSH.wav', 'ShoppingA.wav', 'Burning_Buzzer.wav', 'Slidey101.wav', 'Chunkz.wav']

current_directory = os.path.dirname(os.path.abspath(__file__))
input_dir = current_directory + '/inputs'
output_dir = current_directory + '/outputs'

class MusicMixer(c.Module):
    whitelist = ['generate']

    def __init__(self):
        if os.path.exists(input_dir) == False:
            os.mkdir(input_dir)

        if os.path.exists(output_dir) == False:
            os.mkdir(output_dir)

        if len(os.listdir(input_dir)) == len(file_list):
            return

        print('downloading...')
        print(f'\r{0}/{len(file_list)} files downloaded', end=' ')

        downlaoded = len(os.listdir(input_dir))

        for file in file_list:
            if os.path.exists(input_dir + f'/{file}') == False:
                file_url = f'https://raw.githubusercontent.com/awtsao/MusicMixer/main/inputs/{file}'

                r = requests.get(file_url)

                with open(input_dir + f'/{file}', 'wb') as f:
                    f.write(r.content)

                downlaoded += 1
            print(f'\r{downlaoded}/{len(file_list)} files downloaded', end=' ')
            # sys.stdout.flush()
        print('\ndownload completed.')

    def generate(self, desired_chord = 'C_MAJOR',
                 desired_scale = 'C_MAJOR',
                 desired_generations = 1,
                 desired_crossover_points = 5,
                 desired_mutations = 10,
                 desired_prob = 50):
        '''
        Parameters:
            desired_chord: the chord you deem as a desirable fitness characteristic in the music samples. The string values can range from A_FLAT_MAJOR to G_SHARP_MAJOR

            desired_scale: the scale you deem as a desirable fitness characteristic in the music samples. The string values can range from A_FLAT_MAJOR to G_SHARP_MAJOR

            desired_generations: the number of iterations you would like the genetic algorithm to run for. A value of 1 would select two parents from the initial population and return the recombined offspring.

            desired_crossover_points: the number of crossover points for a sound sample in frequency space.

            desired_mutations: the number of potential mutations you would like to occur for each offspring.

            desired_prob: the percent probability for a single mutation to occur.
        '''
        file_list = os.listdir(input_dir)
        init_pop = []
        rand_idxs = random.sample(range(len(file_list)), int(len(file_list) / 2))

        for i in rand_idxs:
            dat = (wavfile.read(input_dir + '/' +file_list[i]))[1]
            if(dat.ndim == 1):
                dat_mono = dat
            else:
                dat_mono = dat.T[0]
            loudest = np.amax(np.abs(dat_mono))
            init_pop.append(np.float32(dat_mono / loudest))

        pop = evolve(init_pop, desired_generations, desired_chord, desired_scale, output_dir, desired_prob, desired_mutations, desired_crossover_points)
        for i in range(len(pop)):
            wavfile.write(output_dir + '/result_' + str(i) + '.wav', 44100, pop[i])
        
        return f'{output_dir}/result_{random.randint(0, len(pop))}.wav'
    
    def gradio(self):
        with gr.Blocks() as demo:
            with gr.Column():
                with gr.Group():
                    chords = gr.Dropdown(label = 'chord', choices = names, value = 0)
                    scales = gr.Dropdown(label = 'scale', choices = SCALE_NAMES, value = 0)
                    generations = gr.Slider(label = 'generations', minimum = 1, maximum = 1000, value = 1, step = 1)
                    crossover_points = gr.Slider(label = 'crossover points', minimum = 1, maximum = 10, value = 5, step = 1)
                    mutations = gr.Slider(label = 'mutations', minimum = 1, maximum = 50, value = 10, step = 1)
                    prob = gr.Slider(label = 'prob', minimum = 1, maximum = 100, value = 50, step = 1)
                    gen_but = gr.Button("Generate")
                with gr.Group():
                    aud_out = gr.Audio(label = 'result')
                gen_but.click(fn = self.generate, inputs = [chords, scales, generations, crossover_points, mutations, prob], outputs = aud_out)
        demo.launch(quiet=True, share=True)

