import time, threading, pathlib, queue, os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
class _Handler(FileSystemEventHandler):
    def __init__(self,q): self.q=q
    def on_modified(self,e): self._enqueue(e)
    def on_created(self,e):  self._enqueue(e)
    def _enqueue(self,e):
        if not e.is_directory: self.q.put(e.src_path)
def start_watch(doc_folder="docs", delta_fn=None):
    q=queue.Queue()
    obs=Observer(); obs.schedule(_Handler(q),doc_folder,recursive=True); obs.start()
    def consumer():
        seen=set()
        while True:
            path=q.get(); seen.add(path)
            if q.empty():          # debounce burst
                for p in list(seen): delta_fn(p)
                seen.clear()
    threading.Thread(target=consumer,daemon=True).start()
