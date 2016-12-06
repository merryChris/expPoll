from concurrent import futures

import sys
import grpc
import time
import getopt
import threading

from expPoll.core.model import ExpModel
from expPoll.utils.repeatable_timer import RepeatableTimer
import expPoll.protos.platform_pb2 as pb

ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Platform(pb.PlatformServicer):

    TIMER_INTERVAL = 20

    def __init__(self):
        self._data = []
        self._timer = RepeatableTimer(Platform.TIMER_INTERVAL, self._collect)
        self._timer.start()
        self._lock = threading.Lock()

        self._model = ExpModel()

    def _collect(self):
        self._lock.acquire()
        data = self._data[:]
        self._data = []
        self._lock.release()

        self._model.update(data)
        self._timer.restart()

    def Fit(self, request_iterator, context):
        for doc in request_iterator:
            self._lock.acquire()
            self._data.append(doc.content)
            self._lock.release()

        return pb.PlatReply(message='Plat Finished Fit')

def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb.add_PlatformServicer_to_server(Platform(), server)
    server.add_insecure_port(address)
    server.start()

    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    opts, _ = getopt.getopt(sys.argv[1:], '', ['address='])
    address = None
    for opt, val in opts:
        if opt == '--address':
            address = val

    if address: serve(address)
