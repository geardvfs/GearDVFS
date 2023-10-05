import os
import time
import sys
import json
import uuid
import multiprocessing
from datetime import datetime
import flask
from flask import Blueprint
from util_modules.cache import cache

test_api = Blueprint("test_api", __name__)
basic_api = Blueprint("basic_api", __name__)

globals()["DeepDep"] = {}

def io_task(pipe,state):
    count = 0
    while True:
        time.sleep(1)
        rec = pipe.recv()
        state.value = 1
        print("receive pipe message {} from {}".format(rec,os.getpid()))
        pipe.send("initial")
        # if count==0: pipe.send("hello from pipe")

@basic_api.route('/')
def hello_world():
    
    cache.set("test",{"lcd":io_task})
    t1 = datetime.now()
    a = cache.get("test")
    t2 = datetime.now()
    print(a,t2-t1)
    return 'Hello, World! This is a flask demo'

@test_api.route('/Test/client_demo', methods=['GET','POST'])
def predict():
    time.sleep(1)
    print("Receive Test Request after Sleep 1 sec",flush=True)
    if flask.request.method == 'POST':
        print(flask.jsonify(flask.request.data.decode('utf-8')))
        # a = flask.request.data.json().get("freq")
        print("nono",flush=True)
    return flask.jsonify({"nice":1})

@test_api.route('/Test/gen_subprocess')
def gen_proc():
    client_id = uuid.uuid4()
    pipe = multiprocessing.Pipe(duplex=True)
    state = multiprocessing.Value("i",0)
    proc = multiprocessing.Process(target=io_task,args=(pipe[1],state),daemon=True)
    client_data = {"proc":proc,"pipe":pipe[0],"state":state}
    globals()["DeepDep"][client_id] = client_data
    proc.start()
    pipe[0].send("send to subprocess")
    pipe[0].recv()
    print(client_data["state"].value)
    return "client id {} . process {}".format(str(client_id),proc.pid)

@test_api.route('/Test/rm_subprocess')
def rm_proc():
    del_list = []
    for client in list(globals()["DeepDep"]):
        # check status of worker process
        if  not globals()["DeepDep"][client]["proc"].exitcode:
            globals()["DeepDep"][client]["proc"].terminate()
            del_list.append(globals()["DeepDep"][client]["proc"].pid)
        del globals()["DeepDep"][client]
    return "All client connections have been cleared! \n{}".format(del_list)

