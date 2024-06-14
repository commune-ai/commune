# MusicMixer
This is a class project for CS 4701 (AI practicum) in the spring of 2019.

## Instructions to Run
In order to run this project, you must first have the libraries
numpy, scipy, os, and random available for python. You should also have
the submodules scipy.io and scipy.fftpack available. Most importantly,
python should be installed on your machine.

### Params

    **desired_chord** is the chord you deem as a desirable fitness characteristic
    in the music samples. The string values can range from A_FLAT_MAJOR
    to G_SHARP_MAJOR

    **desired_scale** is the scale you deem as a desirable fitness characteristic
    in the music samples. The string values can range from A_FLAT_MAJOR
    to G_SHARP_MAJOR

    **desired_generations** is the number of iterations you would like the
    genetic algorithm to run for. A value of 1 would select two parents from the
    initial population and return the recombined offspring.

    **desired_crossover_points** is the number of crossover points for a
    sound sample in frequency space.

    **desired_mutations** is the number of potential mutations you would like
    to occur for each offspring.

    **desired_prob** is the percent probability for a single mutation to occur.


## Serving the Model

To serve the model with a specific tag and API key, use the following command:

```bash
c model.music_mixer serve tag=latest 
```

## Registering the Model

To register the model with a specific tag and name, use:

```bash
c model.music_mixer register  tag=latest
```

This will set the name to `model.music_mixer::latest`.

## Testing the Module

### Generate

`c model.music_mixer generate'`

Generating random mixed music


### Gradio UI

`c model.music_mixer gradio`

Testing the module on Gradio UI.