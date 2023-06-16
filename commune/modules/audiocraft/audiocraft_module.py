import commune as c
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class AudioCraft(c.Module):
    def __init__(self, model='melody', ,duration=8):
        pass
    
    def set_model(self, config):
        self.model = MusicGen.get_pretrained(self.config.model)
        model.set_generation_params(duration=self.config.duration)  # generate 8 seconds.
    
    
    def generate(self, 
                 descriptons: list[str] = ['happy rock', 'energetic EDM', 'sad jazz'],
                 strategy="loudness",
                 loudness_compressor=True):
        wav = self.model.generate_unconditional(self.config.generations)    # generates 4 unconditional audio samples
        wav =  self.model.generate(descriptons)  # generates 3 samples.
        
        if path:
            path = self.resolve_path(path)
            for idx, one_wav in enumerate(wav):
                audio_write(f'{path}_{idx}', one_wav.cpu(), self.model.sample_rate, strategy=strategy, loudness_compressor=loudness_compressor)
                
    def generate_unconditional(self, generations: int = 4, path: bool = None):
        wav = self.model.generate_unconditional(generations)    # generates 4 unconditional audio samples


    def save_samples(self, wav, path: str = None):


            
        

model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # generates 3 samples.

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

    
    def bro(whadup:int='fam'):
        print(whadup)
        

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    


model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=8)  # generate 8 seconds.
wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
wav = model.generate(descriptions)  # generates 3 samples.

melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)