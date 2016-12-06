#!/usr/bin/env python

import os, sys
from grpc.tools import protoc

def gen_py(protoFile):
    protoc.main(
        (
            '',
            '-I.',
            '--python_out=.',
            '--grpc_python_out=.',
            protoFile,
        )
    )


def gen_go(protoFile):
    protoc.main(
        (
            '',
            '-I.',
            '--go_out=plugins=grpc:%s/src/github.com/merryChris/docDropper/protos' % os.getenv('GOPATH'),
            protoFile,
        )
    )


if __name__ == '__main__':
    if len(sys.argv) == 3:
        sys.argv[1] == 'py' and gen_py(sys.argv[2])
        sys.argv[1] == 'go' and gen_go(sys.argv[2])
