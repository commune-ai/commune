import commune as c

import datetime as dt

import streamlit as st


class StreamlitWatchdog(c.Module):

    @classmethod
    def update_fn(self):
        # Rewrite the dummy.py module. Because this script imports dummy,
        # modifiying dummy.py will cause Streamlit to rerun this script.
        c.print('Updating dummy module')


    @staticmethod
    @st.cache_data
    def install_monitor(path=c.lib_path, recursive=True):
        # Because we use st.cache, this code will be executed only once,
        # so we won't get a new Watchdog thread each time the script runs.

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class Watchdog(FileSystemEventHandler):
            def __init__(self, hook):
                self.hook = hook

            def on_modified(self, event):
                self.hook()

        observer = Observer()
        observer.schedule(
            Watchdog(StreamlitWatchdog.update_fn),
            path=path,
            recursive=recursive)
        observer.start()

        
    
    def install(self):
        c.cmd('pip3 install watchdog')
        c.print('Installing watchdog')
        c.print('Installing watchdog')
        return {'status':'success', 'message':'Installed watchdog'}