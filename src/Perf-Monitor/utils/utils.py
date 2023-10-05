import subprocess
import time
import utils.perf_lib.PyPerf as Perf

def sample(config, monitor, events, cpus, t):
    raw = Perf.sys_perf(cpus, events, int(t))
    log_data = {}
    log_data = monitor.query()
    return log_data, raw


def getAvailableClock(idx=0):
    fname='/sys/devices/system/cpu/cpufreq/policy%s/scaling_available_frequencies' %(idx)
    with open(fname,'r') as f:
        line=f.readline()
        clk_list = [int(s) for s in line.split()]
    return clk_list

def getAvailableClockGPU():
    fname='/sys/devices/gpu.0/devfreq/17000000.gv11b/available_frequencies'
    with open(fname,'r') as f:
        line=f.readline()
        clk_list = [int(s) for s in line.split()]
    return clk_list

# CPU Cores Utilization
def parse_core_util(prev_cpu_time, num_cpu):
    last_idles, last_totals = prev_cpu_time   
    with open('/proc/stat') as f: 
        lines = f.readlines()
    utils = []
    for i, l in enumerate(lines[1:num_cpu+1]):
        fields = [float(column) for column in l.strip().split()[1:]]
        idle, total = fields[3], sum(fields)
        idle_delta, total_delta = idle - last_idles[i], total - last_totals[i]
        last_idles[i], last_totals[i] = idle, total
        utilization = 1.0 - idle_delta / total_delta
        utils.append(utilization)
    return utils, (last_idles, last_totals)

def get_core_time(num_cpu):
    with open('/proc/stat') as f: 
        lines = f.readlines()
    idles,totals = [0]*num_cpu, [0]*num_cpu
    for i, l in enumerate(lines[1:num_cpu+1]):
        fields = [float(column) for column in l.strip().split()[1:]]
        idles[i] = fields[3]
        totals[i] = sum(fields)
    return idles, totals

def check_cpus(config):
    cpu_num = int(config['cpu']['num'])
    online_cpu_num = 0
    for i in range(cpu_num):
        if get_value(config['cpu']['on'].replace("$$",str(i)))=="1":
            online_cpu_num += 1
    return online_cpu_num

# File operations for sysfs nodes
def get_value(file):
    with open(file, 'r') as f:
        text = f.read().strip("\n")
    return text

def read_value(f):
    f.seek(0)
    text = f.read().strip("\n")
    return text

def set_value(file,v):
    with open(file,'w') as f:
        f.write(str(v))
    return 0

def s2i(s):
    # convert string to int
    i = int(s.replace(',',''))
    return i