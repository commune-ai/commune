import http
import commune as c
import gradio as gr

class Stock(c.Module):
    
    whitelist = ['forward', 'ask', 'generate', 'call']

    def __init__(self, api_key:str = None, host='api.polygon.io', cache_key:bool = True):
        self.set_config(kwargs=locals())
        self.conn = http.client.HTTPSConnection(self.config.host)
        self.set_api_key(api_key=self.config.api_key, cache=self.config.cache_key)

    def set_api_key(self, api_key:str, cache:bool = True):
        if api_key == None:
            api_key = self.get_api_key()

        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)

    def call(self,
             api_key:str = None,    # API key from polygon-io
             ticker:str = 'AAPL',   # The ticker symbol of the stock/equity.
             multiplier: int = 1,   # The size of the timespan multiplier.
             timespan: str = 'day', # The size of the time window.
             start: str = '2023-01-01', # The start of the aggregate time window. Either a date with the format YYYY-MM-DD or a millisecond timestamp.
             end: str = '2023-01-02',   # The end of the aggregate time window. Either a date with the format YYYY-MM-DD or a millisecond timestamp.
             adjusted: bool = True, # Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
             sort: str = 'asc', # Sort the results by timestamp. `asc` will return results in ascending order (oldest at the top), `desc` will return results in descending order (newest at the top).
             limit: int = 5000  # Limits the number of base aggregates queried to create the aggregate results. Max 50000 and Default 5000.
    ) -> str:
        api_key = api_key if api_key != None else self.api_key
        payload = ''
        headers = {}
        self.conn.request("GET", f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}?adjusted={adjusted}&sort={sort}&limit={limit}&apiKey={api_key}", payload, headers)
        res = self.conn.getresponse()
        data = res.read()
        return data.decode("utf-8")
    def gradio(self):
        with gr.Blocks(title="Stock") as demo:
            with gr.Column():
                with gr.Row():                    
                    api_key = gr.Text(label="api_key", type='password', interactive=True)
                    ticker = gr.Text(label="Ticker", value="AAPL", interactive=True)
                with gr.Row():
                    start_date = gr.Text(label="Start Date", value="2023-01-01", interactive=True)
                    end_date = gr.Text(label="End Date", value="2023-01-20", interactive=True)
                test_btn = gr.Button()
                output = gr.Text(label="Result")
            
            def generate(api_key, ticker, start_date, end_date):
                return gr.update(value=self.call(api_key, ticker, start=start_date, end=end_date))
            test_btn.click(fn = generate, inputs = [api_key, ticker, start_date, end_date], outputs = [output])
            
        demo.launch(share=True)
        # demo.launch()

    forward = ask = generate = call
    
    ## API MANAGEMENT ##

    @classmethod
    def add_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}
    
    @classmethod
    def rm_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   

        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def get_api_key(cls):
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return c.choice(api_keys)

    @classmethod
    def api_keys(cls):
        return cls.get('api_keys', [])
    
    