import commune

class Launchpad(commune.Module):
    
    @classmethod
    def models(cls, model = 'gptj'):
        '''
        ArXiv/            Gutenberg_PG/
        BookCorpus2/      HackerNews/
        Books3/           NIHExPorter/
        DMMathematics/    OpenSubtitles/
        '''
        model_module = commune.get_module('model.transformer')
        datasets = ['ArXiv', 'Gutenberg_PG', 'BookCorpus2', 'HackerNews', 'Books3', 'NIHExPorter', 'DMMathematics', 'OpenSubtitles']
        import time
        for i in range(4):
            model_module.pm2_kill(name=f'model::{model}::{i}::{i}')
            model_module.launch(name=f'model::{model}', kwargs={'model': model, 'tag': str(i)})


if __name__ == "__main__":
    Launchpad.run()