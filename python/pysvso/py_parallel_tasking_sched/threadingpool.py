import os
import sys

# import threading libraries

import threading
# useful thread library
import ctypes

LIBC = ctypes.cdll.LoadLibrary('libc.so.6')

# instruction number, see implementation of linux kernel /usr/include/x86_64-linux-gnu/asm/unistd_64.h
__NR_gettid = 186


def SelfThread():
    return LIBC.syscall(__NR_gettid)

# SelfThread()
try:
    import queue
except:
    import Queue as queue
import logging
from pysvso.lib.log import LoggerAdaptor
# create stream handler and add it to root logger, otherwise your logging in the colab won't work
console = logging.StreamHandler(stream=sys.stdout)

root = logging.getLogger()
for handler in root.handlers:
    root.removeHandler(handler)

fmt = logging.Formatter("%(asctime)s [%(levelname)s]:%(filename)s.%(name)s, in line %(lineno)s >> %(message)s")
console.setFormatter(fmt)
console.setLevel(logging.INFO)
root.addHandler(console)

_logger = logging.getLogger("ThreadingPool")

from pysvso.lib.misc import AtomicCounter

# import async libraries
LogAdaptor = LoggerAdaptor

class Worker(threading.Thread):
    logger = LogAdaptor("Worker", _logger)
    Seq = AtomicCounter()

    def __init__(self, tasks, name="worker"):
        # identity
        self.seq = self.Seq()
        self.tid = None

        threading.Thread.__init__(self, name=name)
        self.tasks = tasks
        self.daemon = True

        self._lock = threading.Lock()
        # implement event loop interface
        # close event
        self._stopped = False

        # stop event
        self._stopping = False

        print("[Thread %d], Starting ..." % (self.seq,))
        self.start()

    def set_stopping(self):
        # might be set from multiple locations
        print("[Thread %d], setting to stop." % (self.seq,))
        with self._lock:
            self._stopping = True
            print("[Thread %d], set to stop." % (self.seq,))

    def isStopping(self):
        return self._stopping

    def run_forever(self):
        self.tid = SelfThread()
        while not self.isStopping():
            self._run_once()
        self._stopped = True
        print("[Thread %d] (system tid %d) stopped." % (self.seq, self.tid))

    def _run_once(self):
        func, args, kw = self.tasks.get()
        try:
            func(*args, **kw)
        except RuntimeErr as e:
            self.logger.error(e)
        finally:
            self.tasks.task_done()

    # for asynchronous events implements linux epoll based _run_once and run_untile_complete
    def run(self):
        self.run_forever()


# used when a thread is working on both I/O and cpu bunded computation only
# AsyncWorker implements logics to interact with Kernel based on Linux epoll based on Asyncio
# Other async machenisims will be supported soon
class AsyncWorker:
    pass


class ThreadingPool:
    def __init__(self, max_workers=5):
        self.tasks = queue.Queue(max_workers)
        self.workers = []
        for _ in range(max_workers):
            self.workers.append(Worker(self.tasks))

    def add_task(self, func, *args, **kw):
        self.tasks.put((func, args, kw))

    def wait_completion(self):
        self.tasks.join()

    def run(self):
        raise Exception("Not Implemented!")


class RuntimeErr(Exception):
    pass