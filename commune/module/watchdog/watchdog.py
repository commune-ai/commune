import commune as c
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, module):
        super().__init__()
        self.module = module


    def on_any_event(self, event):
        if event.is_directory:
            return
        if event.event_type in ['created', 'modified', 'deleted']:
            c.print(f'File change detected: {event.src_path}')
            c.module_tree(update=True, verbose=True)

class WatchdogModule(c.Module, FileSystemEventHandler):



    def __init__(self, folder_path:str = c.root_path, run:bool = True ):
        super().__init__()
        self.folder_path = folder_path
        self.observer = None
        if run:
            c.thread(self.start_server)
    def start_server(self):
        event_handler = FileChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.folder_path, recursive=True)
        self.observer.start()

        try:
            lifetime = 0
            sleep_period = 5
            while True:
                c.print(f'Watching for file changes. {lifetime} seconds elapsed.')
                
                time.sleep(sleep_period)
                lifetime += sleep_period
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

    def stop_server(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None

    def log_file_change(self, message):
        c.print(message)
