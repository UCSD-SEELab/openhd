#!/usr/bin/env python
import logging
import sys
from subprocess import Popen, PIPE

from py14.transpiler import transpile
from flask import Flask, request


app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def static_proxy(path):
    return app.send_static_file(path)


@app.route('/transpile', methods=['POST'])
def transpilation():
    source = request.data
    print(request.data)
    cpp = transpile(source, headers=False, testing=False)
    return format_cpp(cpp), 200, {'Content-Type': 'text/plain'}


def format_cpp(cpp):
    proc = Popen(['clang-format-3.5'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    formatted_cpp = proc.communicate(input=cpp)[0]
    proc.wait()
    return formatted_cpp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
