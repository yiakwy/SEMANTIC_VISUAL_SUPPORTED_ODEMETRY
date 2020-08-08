import threading
# used to generate uuid
import uuid

class AtomicCounter(object):

    def __init__(self):
        self._counter = 0
        self.lock = threading.Lock()

    def incr(self):
        with self.lock:
            self._counter += 1
            return self._counter

    def __call__(self):
        return self.incr()


# @todo : TODO implements various kinds authorization verification codes inside an identifier
class Identity:
    def __init__(self, seq, name=None, id=None, uuid=None, tok=None):
        self.seq = seq
        self.name = name
        self.id = id
        self.uuid = uuid
        self.tok = tok

    def __str__(self):
        return "Identity#%d" % self.seq