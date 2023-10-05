import time
import sys
import os
import json
import multiprocessing
import uuid

from graph.learners import dqn_nx   

import flask
from flask import Blueprint

nn_api = Blueprint("nn_api", __name__)

"""Register global variable to store in memeroy models
Format:{"DeepDep":{
    Client1<uuid4>: {"proc":<Process>,"pipe":<Pipe>,"status":<Bool>,"m_type":<str>},
        ...
    },
}}
"""
globals()["DeepDep"] = {}

@nn_api.route('/nn/test_action', methods=['POST'])
def test_action():
    init_params = json.loads(flask.request.json)
    m_type = init_params["m_type"]
    learner = learner_factory(m_type)
    return flask.jsonify({"status":False,"result":learner()})

@nn_api.route('/nn/init_model', methods=['POST'])
def init_model():
    init_params = json.loads(flask.request.json)
    m_type = init_params["m_type"]
    learner = learner_factory(m_type)
    if learner: 
        client_id = uuid.uuid4()
        pipe,state = multiprocessing.Pipe(duplex=True),multiprocessing.Value("i",0)
        proc = multiprocessing.Process(
            target=learner,args=(pipe[1],state,init_params["params"]),daemon=True)
        proc.start()
        if pipe[0].recv(): # Todo: set timeout limit/try catch
            globals()["DeepDep"][client_id] = {
                "state":state,"m_type":m_type,"pipe":pipe[0],"proc":proc
            }
            return flask.jsonify({"status":True,"m_id":str(client_id)})
    return flask.jsonify({"status":False})

@nn_api.route('/nn/rm_model', methods=['GET'])
def rm_model():
    status = rm_client(flask.request.args.get("m_id"))
    return flask.jsonify({"status":status})

@nn_api.route('/nn/clear_all')
def clear_all():
    del_list = []
    for client in list(globals()["DeepDep"]):
        rm_client(str(client))
    return "All client connections have been cleared! \n{}".format(del_list)

@nn_api.route('/nn/get_action', methods=['POST'])
def get_action():   
    params = json.loads(flask.request.json)
    m_id, data = params["m_id"], params["data"]
    # ,params["s"]
    client_id = check_client(m_id)
    if client_id:
        learner = globals()["DeepDep"][client_id]
        print(learner['state'].value == 1)
        learner['pipe'].send({"cmd":"RECORD","data":data})
        a = learner['pipe'].recv()
        return flask.jsonify({"status":True,"action":a,"max_val":a})
    else:
        return flask.jsonify({"status":False})

@nn_api.route('/nn/get_action_test', methods=['POST'])
def get_action_test():   
    params = json.loads(flask.request.json)
    m_id, data = params["m_id"], params["data"]
    # ,params["s"]
    client_id = check_client(m_id)
    if client_id:
        learner = globals()["DeepDep"][client_id]
        print(learner['state'].value == 1)
        learner['pipe'].send({"cmd":"TEST","data":data})
        a = learner['pipe'].recv()
        return flask.jsonify({"status":True,"action":a,"max_val":a})
    else:
        return flask.jsonify({"status":False})

@nn_api.route('/nn/get_test_power', methods=['POST'])
def get_test_power():   
    params = json.loads(flask.request.json)
    m_id = params["m_id"]

    client_id = check_client(m_id)
    if client_id:
        learner = globals()["DeepDep"][client_id]
        learner['pipe'].send({"cmd":"END_TEST"})
        p = learner['pipe'].recv()
        return flask.jsonify({"status":True,"result":p})
    else:
        return flask.jsonify({"status":False})

@nn_api.route('/nn/train', methods=['POST'])
def get_action_update():
    params = json.loads(flask.request.json)
    m_type, m_id, t = params["m_type"],params["m_id"],params["t"]
    m_uuid = check_client(m_id)
    if m_uuid:
        learner = globals()["DeepDep"][m_type][m_uuid]
        s,s1,a,r = t["s"],t["s1"],t["a"],t["r"]
        a_new, max_val = learner.get_action(s,eps=params['eps'])
        learner.update_table((s,a,s1,r))
        return flask.jsonify({"status":True,"action":a_new,"max_val":max_val})
    else:
        return flask.jsonify({"status":False})

@nn_api.route('/nn/check_model_status', methods=['GET'])
def check_model_status():
    client_id = check_client(flask.request.args.get("m_id"))
    if client_id:
        state = globals()["DeepDep"][client_id]['state'].value == 1
        if state: return flask.jsonify({"status":True})
    return flask.jsonify({"status":False})

@nn_api.route('/nn/request_update', methods=['GET'])
def request_update():
    client_id = check_client(flask.request.args.get("m_id"))
    if client_id:
        learner = globals()["DeepDep"][client_id]
        learner['pipe'].send({"cmd":"TRAIN"})
        return flask.jsonify({"status":True})
    return flask.jsonify({"status":False})

"""
Support Functions
"""
def check_client(str_id):
    try:
        client_id = uuid.UUID(str_id)
        t = globals()["DeepDep"][client_id]
        return client_id
    except Exception:
        return None

def rm_client(str_id):
    client_id = check_client(str_id)
    if client_id:
        # check status of worker process
        if not globals()["DeepDep"][client_id]['proc'].exitcode:
            globals()["DeepDep"][client_id]['proc'].terminate()
        del globals()["DeepDep"][client_id]
        return True
    return False

"""
learners
"""
def mock_learner():
    return "mock_learner"



def learner_factory(m_type):
    learner_dict = {
        "MOCK":mock_learner,
        "DQN":dqn_nx
    }
    learner = learner_dict.get(m_type, None)
    return learner