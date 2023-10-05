from utils.utils import *
import time

'''
Monitoring Features
    Process Variables:
        - CPU Utilization
    State Variables:
        - Power
        - Temperature
        - Frequency(optional)
'''

class WindowAverageMeter(object):
    def __init__(self):
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Monitor(object):
    def __init__(self, config):
        self.num_cpu = check_cpus(config)
        print("Monitor Process Started for {}".format(config["device"]["name"]))
        print("Num of online cpus: {}".format(self.num_cpu))

        self.powers, self.thermals = [[]]*2

        for i, k in enumerate(config['power']):
            node = config['power'][k]
            self.powers.append({
                "node":node,"f":open(node,'r'),"name":k,"m":AverageMeter()})

        for i in range(2):
            node = config['thermal']['temp'].replace("$$",str(i))
            name = get_value(config['thermal']['t_type'].replace("$$",str(i)))
            self.thermals.append({
                "node":node,"f":open(node,'r'),"name":name,"m":AverageMeter()})

        # self.prev_cpu_time = get_core_time(self.num_cpu)
        self.gpu_util_node = open(config['gpu']['load'],'r')
        # frequencies
        self.gpu_freq_node = open(config['gpu']['freq'],'r')
        self.cpu_freq_node = open(config['cpu']['freq'].replace("$$",str(0)),'r')

    def __sample(self):
        # Sample Powers
        for domain in [self.powers, self.thermals]:
            for item in domain:
                val = read_value(item['f'])
                item["m"].update(float(val))

    def reset(self):
        for domain in [self.powers, self.thermals]:
            for item in domain: item["m"].reset()
        # self.prev_cpu_time = get_core_time(self.num_cpu)
        self.__sample()

    def query(self):
        # utils, self.prev_cpu_time = parse_core_util(self.prev_cpu_time,self.num_cpu)
        self.__sample()
        query_result = {}
        for domain in [self.powers, self.thermals]:
            for item in domain:
                query_result[item["name"]] = item["m"].avg
        query_result["gpu_util"] = float(read_value(self.gpu_util_node))/1000
        query_result["gpu_f"] = float(read_value(self.gpu_freq_node))
        query_result["cpu_f"] = float(read_value(self.cpu_freq_node))
        return query_result


def monitor_daemon(config,conn):
    monitor = Monitor(config)
    while True:
        pass