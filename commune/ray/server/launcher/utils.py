class CronJob:
    def __init__(self,cfg):
        self.last_run_timestamp = self.current_timestamp
        self.cfg = cfg
        self.interval = cfg['interval']
        self.job_kwargs = self.cfg['job']
        self.expiration = cfg.get('expiration', self.current_timestamp + 3600)

    @property
    def current_timestamp(self):
        return datetime.datetime.utcnow().timestamp()


    @property
    def last_run_delay(self):
        return (self.current_timestamp - self.last_run_timestamp)

    @property
    def should_update(self):
        return  self.last_run_delay > self.cfg['interval']

    def get_job_kwargs(self):
        if self.expired:
            return None


        if self.should_update:
            self.last_run_timestamp = self.current_timestamp
            return self.job_kwargs
        else: 
            return None
    
    @property
    def expired(self):
        return self.current_timestamp > self.expiration
        
