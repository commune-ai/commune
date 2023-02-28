import commune

class Launchpad(commune.Module):
    
    
    @classmethod
    def train(cls, models=[f'model::gptj::{i}' for i in [0]], 
              datasets=['dataset::bittensor']):
        
        model_module = commune.get_module('model.transformer')
        for model in models:
            for dataset in datasets:
                model_module.launch(fn='remote_train',name=f'train::{model}', kwargs={'model': model, 'dataset': dataset, 'save': True}, serve=False)
    @classmethod
    def models(cls, models = ['gptj', 'gpt3b']):
        '''
        ArXiv/            Gutenberg_PG/
        BookCorpus2/      HackerNews/
        Books3/           NIHExPorter/
        DMMathematics/    OpenSubtitles/
        '''
        model_module = commune.get_module('model.transformer')
        model_module.launch(name=f'model::{model}', kwargs={'model': model, 'tag':str(i)})
        # datasets = ['ArXiv', 'Gutenberg_PG', 'BookCorpus2', 'HackerNews', 'Books3', 'NIHExPorter', 'DMMathematics', 'OpenSubtitles']
        # import time
        # for model in models:
        #     for i in range(4):
        #         # model_module.pm2_kill(name=f'model::{model}')
        #         model_module.launch(name=f'model::{model}', tag=str(i), kwargs={'model': model})


if __name__ == "__main__":
    Launchpad.run()