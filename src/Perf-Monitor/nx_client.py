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
AVAIL_GOVS_GPU = ["simple_ondemand","userspace","performance", "nvhost_podgov"]
# AVAIL Benchmarks
AVAIL_BENCHS = {
    "video":"LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lcd/ffmpeg_build/lib python3 ./benchmarks/render/player.py",
    "yolo":None,
    "cpu":"sysbench --test=cpu --num-threads=3 --cpu-max-prime=20000 run",
    "mem":"sysbench --test=memory --num-threads=1 --memory-block-size=2G --memory-total-size=100G run",
    "io":'sysbench --threads=16 --events=10000 fileio --file-total-size=3G --file-test-mode=rndrw run',
    "thread":'sysbench threads --time=5 --thread-yields=2000 --thread-locks=2 --num-threads=64  run',
    }
# There are also some hw cache events
AVAIL_EVENTS = ["cycles","instructions","cache-misses","cache-references"]
# AVAIL FREQS
AVAIL_FREQS = [i*100000 for i in range(2,21)]

# time in us(microseconds)
PERF_TIME = 10*1e3

TARGET_GOV = AVAIL_GOVS[2]
TARGET_GOV_GPU = AVAIL_GOVS_GPU[0]
TARGET_BENCH = AVAIL_BENCHS["io"]
TARGET_BENCH = "sleep 60"
BENCH_EPOCH, TEST_EPOCH = 2, 3
def main():
    # Load Configurations
    device = "./configs/nx"
    config = configparser.ConfigParser()
    config.read("{}.ini".format(device))

    # Determine Perf Events
    cpus = [0,1,2,3,4,5]
    events = [0,4,5] # ["cycles","stalled-cycles-front", "stalled-cycles-back"]

    # Switch cpufreq governor
    set_value(config["cpu"]["gov"].replace("$$",str(0)),TARGET_GOV)
    AVAIL_CPU_FREQS = getAvailableClock(idx=0)
    # Switch gpu devfreq governor
    set_value(config['gpu']['gov'], TARGET_GOV_GPU)
    AVAIL_GPU_FREQS = getAvailableClockGPU()
    print(AVAIL_GPU_FREQS)
    print(AVAIL_FREQS)
    # Create monitor
    monitor = Monitor(config) 

    count, start_time = 0, time.time()
    # Benchmark training loop
    for epoch in range(1, BENCH_EPOCH):
        TRAIN = False if epoch%TEST_EPOCH==0 else True
        # Start bechmark process
        bench_proc = subprocess.Popen(TARGET_BENCH,shell=True)
        # Sampling loop
        while(True):
            # Reset Monitor
            t = time.time()
            # Run perf cmd & Request monitor data
            try:
                monitor.reset()
                log_data, pmus = sample(config, monitor, events, cpus, PERF_TIME)
                print(log_data, pmus)
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
            b_state = subprocess.Popen.poll(bench_proc)
            if b_state == None:
                time.sleep(0)
            elif b_state == 0:
                if TRAIN:
                    print("Train Epoch({})End of Benchmarking  with {} perf records".format(epoch, count))
                else:
                    resp = get_test_power(url_base,m_info)['result']
                    print("Test Result: Power {} and Time {}".format(resp["p"],resp["t"]))
                break
    # rm_model(url_base,m_info)    

    # Reset to default governors
    set_value(config["cpu"]["gov"].replace("$$",str(0)),'schedutil')
    set_value(config['gpu']['gov'], 'simple_ondemand')

if __name__ == '__main__':
    main()
