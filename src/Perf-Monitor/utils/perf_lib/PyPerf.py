import sys
import os
import os.path as path
from ctypes import *


print(path.join(path.abspath(path.dirname(__file__)),"sys_perf.so"))
_lib = CDLL(path.join(path.abspath(path.dirname(__file__)),"sys_perf.so"))



_lib.get_supported_events.restype = py_object
def get_supported_events():
    events = _lib.get_supported_events()
    return events

_lib.get_supported_abbrevs.restype = py_object
def get_supported_abbrevs():
    abbrevs = _lib.get_supported_abbrevs()
    return abbrevs

_lib.get_supported_names.restype = py_object
def get_supported_names():
    names = _lib.get_supported_names()
    return names

_lib.sys_perf.restype = py_object
_lib.sys_perf.argtypes = (py_object, py_object,c_int)
def sys_perf(cpus, events, ms):
    raw_pmus = _lib.sys_perf(cpus, events, ms)
    return raw_pmus


if __name__ == '__main__':
    cpus = [0,1]
    events = [0]
    abbrevs = get_supported_abbrevs()
    pmus = sys_perf(cpus,events,1000000)
    result_dict = {}
    for i, c in enumerate(cpus):
        for j,e in enumerate(events):
            name = "cpu{}_{}".format(c,abbrevs[e])
            result_dict[name] = pmus[i][j]
    print(result_dict)
