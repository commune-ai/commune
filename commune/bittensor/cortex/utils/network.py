class IOCounter:

    def __init__(self, mode='mb'): 
        assert mode in ['b', 'mb', 'gb']  
        self.mode = mode
        # self.__dict__.update(locals())
        
    def __enter__(self):
        io_2 = psutil.net_io_counters()
        self.initial_upload_bytes, self.initial_download_bytes = io_2.bytes_sent, io_2.bytes_recv

    def __exit__(self):
        io_2 = psutil.net_io_counters()
        self.total_upload_bytes, self.total_download_bytes = io_2.bytes_sent-self.initial_upload_bytes, io_2.bytes_recv - self.initial_download_bytes

    # def 
