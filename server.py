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

    TIMER_INTERVAL = 200

    def __init__(self, cool=True):
        self.data = []

        self.timer = RepeatableTimer(Platform.TIMER_INTERVAL, self._collect)
        self.timer.start()
        self.lock = threading.Lock()
        self.model = ExpModel(not cool)

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

            doc = copy.deepcopy(req.title)
            for _ in range(10): doc.MergeFrom(req.title)
            doc.MergeFrom(req.content)
            self.data.append(doc)

            self.lock.release()

        return pb.CommonResponse(code=0, message='Goodbye from platform Server')

    def Filter(self, request, context):
        ready, res = self.model.Pick(request.tokens)
        if not ready: return pb.FilterResponse(code=1, tokens=[])

        return pb.FilterResponse(code=0, tokens=res)

    def Query(self, request, context):
        ready, ctx = self.model.Expand(request.keywords)
        if not ready: return pb.QueryResponse(code=1, keywords=[], probabilities=[])

        k, p = [None]*len(ctx), [None]*len(ctx)
        for i in range(len(ctx)):
            k[i], p[i] = ctx[i]

        return pb.QueryResponse(code=0, keywords=k, probabilities=p)

def serve(address, cool=True):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb.add_PlatformServicer_to_server(Platform(cool), server)
    server.add_insecure_port(address)
    server.start()

    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    opts, _ = getopt.getopt(sys.argv[1:], '', ['address=', 'cool='])
    address, cool = None, True
    for opt, val in opts:
        if opt == '--address':
            address = val
        if opt == '--cool':
            cool = val == 'True' and True or False

    if address: serve(address, cool)
