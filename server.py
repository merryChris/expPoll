from concurrent import futures

import sys
import copy
import grpc
import time
import getopt
import threading

from expPoll.core.model import ExpModel
from expPoll.utils.repeatable_timer import RepeatableTimer
import expPoll.protos.platform_pb2 as pb

ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Platform(pb.PlatformServicer):

    TIMER_INTERVAL = 100

    def __init__(self):
        self.data = []
        self.docDic = {}

        self.timer = RepeatableTimer(Platform.TIMER_INTERVAL, self._collect)
        self.timer.start()
        self.lock = threading.Lock()
        self.model = ExpModel()

    def _collect(self):
        self.lock.acquire()
        saved_data = self.data[:]
        del self.data[:]
        self.lock.release()

        self.model.Update(saved_data)
        self.timer.restart()

    def Fit(self, request_iterator, context):
        for req in request_iterator:
            self.lock.acquire()

            self.docDic[len(self.docDic)] = req.hash
            doc = copy.deepcopy(req.title)
            for _ in range(10): doc.MergeFrom(req.title)
            doc.MergeFrom(req.content)
            self.data.append(doc)

            self.lock.release()

        return pb.PlatReply(message='Goodbye from platform Server')

    def Query(self, request, context):
        ctx = self.model.Expand(request.keywords)
        docs = self.model.Seek(ctx) if ctx else []

        return pb.QueryResponse(hashs=[self.docDic[d[0]] for d in docs])

def serve(address):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
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
