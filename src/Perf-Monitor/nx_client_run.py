import os
import os.path as path
import time
import datetime
import random
import configparser
import subprocess
from multiprocessing import Process, Pipe

from utils.utils import *
from utils.nn_api import *
from utils.monitor import Monitor

# Device specific constants
AVAIL_GOVS = ["ondemand","schedutil","userspace","conservative","powersave","performance"]

# There are also some hw cache events
AVAIL_EVENTS = ["cycles","instructions","cache-misses","cache-references"]
# AVAIL FREQS
AVAIL_FREQS = [i*100000 for i in range(2,21)]

PERF_TIME = 0.025

TARGET_GOV = AVAIL_GOVS[1]


def main():
    # Load Configurations
    device = "./configs/nx"
    config = configparser.ConfigParser()
    config.read("{}.ini".format(device))

    # Determine Perf Events
    perf_path,events = "/usr/bin/perf", ["cycles","instructions"]

    # Switch cpufreq governor
    set_value(config["cpu"]["gov"].replace("$$",str(0)),TARGET_GOV)
    # todo: Switch gpu devfreq governor

    # Create monitor
    monitor = Monitor(config) 

    count, start_time = 0, time.time()
    # Benchmark training loop
    for epoch in range(1, BENCH_EPOCH):

        TRAIN = False if epoch%TEST_EPOCH==0 else True
        # Sampling loop
        while(True):
            # Reset Monitor
            t = time.time()
            # Run perf cmd & Request monitor data
            try:
                monitor.reset()
                log_data = sample(config,monitor)
                # log_data = step_sample(config,start_time,perf_cmd,parent_conn)
            except Exception as err:
                print("Perf Failed with Err: {}".format(err))
                return

            print(time.time()-t)
            # state_params = {"data":log_data}
            # if TRAIN:
            #     # Request action from NN Server
            #     resp = get_action(url_base,m_info,state_params)
            #     req_idx = resp['action']['action']
            #     # Set new frequency
            #     set_value(config["cpu"]["freq_u"].replace("$$",str(0)),AVAIL_FREQS[freq_idx])
            #     count += 1
            #     if count%TRAIN_STEP == 0:
            #         if request_update(url_base, m_info)['status']
            #             # Poll until train step
            #             while True:
            #                 time.sleep(1)
            #                 if check_model_status(url_base,m_info)['status']: break
            # else:
            #     # Inference only
            #     resp = get_action_test(url_base,m_info,state_params)
            #     freq_idx = resp["action"]["action"]
            #     # Set new frequency
            #     set_value(config["cpu"]["freq_u"].replace("$$",str(0)),AVAIL_FREQS[freq_idx])

            # Check state of benchmark process

    # rm_model(url_base,m_info)            
    # Reset to default governors
    set_value(config["cpu"]["gov"].replace("$$",str(0)),'schedutil')

if __name__ == '__main__':
    main()
