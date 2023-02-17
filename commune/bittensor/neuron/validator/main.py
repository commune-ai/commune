from neuron import neuron
import bittensor
if __name__ == "__main__":
    bittensor.utils.version_checking()
    neuron().run()