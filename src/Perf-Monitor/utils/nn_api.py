import requests
import json

def test_action(url_base, m_info, init_params):
    data = {**m_info, "params":{**init_params}}
    url = url_base + "/test_action"
    resp = requests.post(url, json=json.dumps(data))
    return json.loads(resp.text)

def init_model(url_base, m_info, init_params):
    data = {**m_info, "params":{**init_params}}
    url = url_base + "/init_model"
    resp = requests.post(url, json=json.dumps(data))
    return json.loads(resp.text)

def rm_model(url_base,m_info):
    url = url_base + "/rm_model"
    resp = requests.get(url, params=m_info)
    return json.loads(resp.text)

def save_model(url_base,m_info):
    url = url_base + "/save_model"
    resp = requests.get(url, params=m_info)
    return json.loads(resp.text)

def get_action(url_base,m_info,params):
    data = {**m_info, **params}
    url = url_base + "/get_action"
    resp = requests.post(url, json=json.dumps(data))
    return json.loads(resp.text)

def get_action_test(url_base,m_info,params):
    data = {**m_info, **params}
    url = url_base + "/get_action_test"
    resp = requests.post(url, json=json.dumps(data))
    return json.loads(resp.text)

def get_test_power(url_base,m_info):
    data = {**m_info}
    url = url_base + "/get_test_power"
    resp = requests.post(url, json=json.dumps(data))
    return json.loads(resp.text)

def request_update(url_base,m_info):
    url = url_base + "/request_update"
    resp = requests.get(url, params=m_info)
    return json.loads(resp.text)

def check_model_status(url_base,m_info):
    url = url_base + "/check_model_status"
    resp = requests.get(url, params=m_info)
    return json.loads(resp.text)
