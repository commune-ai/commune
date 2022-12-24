
import psutil

class NetworkMonitor:

    def __enter__(self):
        io_1 = psutil.net_io_counters()
        self.start_bytes_sent, self.start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv
        self.start_time = datetime.datetime.utcnow()
        return self

    def __exit__(self, *args):
        io_2 = psutil.net_io_counters()
        self.total_upload_bytes, self.total_download_bytes = self.upload_bytes, self.download_bytes
    
    @property
    def download_bytes(self):
        io_2 = psutil.net_io_counters()
        return io_2.bytes_sent - self.start_bytes_sent

    @property
    def upload_bytes(self):
        io_2 = psutil.net_io_counters()
        return io_2.bytes_recv - self.start_bytes_recv



