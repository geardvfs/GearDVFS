#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <Python.h>

typedef struct{
    char* name;
    char* abbrev;
    int val;
} PerfEvent;

struct read_format {
  uint64_t nr;
  struct {
    uint64_t value;
    uint64_t id;
  } values[];
};

const PerfEvent EVENT_LIST[] = {
    {"PERF_COUNT_HW_CPU_CYCLES", "cycles", PERF_COUNT_HW_CPU_CYCLES},
    {"PERF_COUNT_HW_INSTRUCTIONS", "instructions", PERF_COUNT_HW_INSTRUCTIONS},
    {"PERF_COUNT_HW_CACHE_REFERENCES",  "cache-ref", PERF_COUNT_HW_CACHE_REFERENCES},
    {"PERF_COUNT_HW_CACHE_MISSES", "cache-miss", PERF_COUNT_HW_CACHE_MISSES},
    {"PERF_COUNT_HW_STALLED_CYCLES_FRONTEND", "stalled-cycles-front", PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
    {"PERF_COUNT_HW_STALLED_CYCLES_BACKEND", "stalled-cycles-back", PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
    {"PERF_COUNT_HW_BRANCH_MISSES", "branch-miss", PERF_COUNT_HW_BRANCH_MISSES},
};

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
               int cpu, int group_fd, unsigned long flags)
{
   int ret;
   ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                  group_fd, flags);
   return ret;
}

PyObject*
sys_perf(PyObject* cpus, PyObject* events, const int micro_seconds)
{   

    Py_ssize_t n_event = PyList_Size(events);
    Py_ssize_t n_cpu = PyList_Size(cpus);
    int N_COUNTER = n_cpu*n_event;
    PyObject* result_list = PyList_New(n_cpu);

    // initialization
    struct perf_event_attr pea;
    struct read_format **rfs = (struct read_format**)calloc(n_cpu, sizeof(struct read_format*));
    int **fds = (int **)calloc(n_cpu, sizeof(int *));
    uint64_t **ids = (uint64_t **)calloc(n_cpu, sizeof(uint64_t *));
    char **bufs = (char **)calloc(n_cpu, sizeof(char *));

    for (int i=0; i < n_cpu; i++) {
        fds[i] = (int*)calloc(N_COUNTER,sizeof(int));
        ids[i] = (uint64_t*)calloc(N_COUNTER,sizeof(uint64_t));
        bufs[i] = (char*)calloc(4096,sizeof(char));
        rfs[i] = (struct read_format*) bufs[i];
    }

    // for each cpu and each hardware event
    for (int i = 0; i < n_cpu; i++) {
        int cpu_index = (int)PyLong_AsLong(PyList_GetItem(cpus,i));
        for (int j = 0; j < n_event; j++) {
            int event_index = (int)PyLong_AsLong(PyList_GetItem(events,j));
            memset(&pea, 0, sizeof(struct perf_event_attr));
            pea.type = PERF_TYPE_HARDWARE;
            pea.size = sizeof(struct perf_event_attr);
            pea.config = EVENT_LIST[event_index].val;
            pea.disabled = 1;
            pea.exclude_kernel = 0;
            pea.exclude_hv = 1;
            pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
            if (j == 0) {
                fds[i][j] = syscall(__NR_perf_event_open, &pea, -1, cpu_index, -1, 0);
                // fprintf(stderr,"%d,%d,%d\n",i,j,fds[i][j]);
            } else {
                fds[i][j] = syscall(__NR_perf_event_open, &pea, -1, cpu_index, fds[i][0], 0);
            }
            if (fds[i][j] == -1) {
                fprintf(stderr,"Error opening leader %llx\n", pea.config);
                exit(EXIT_FAILURE);
            }
            ioctl(fds[i][j], PERF_EVENT_IOC_ID, &ids[i][j]);
        }
    }

    // monitoring for each cpu group
    for (int i=0; i < n_cpu; i++) {
        ioctl(fds[i][0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fds[i][0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);   
    }
    usleep(micro_seconds);
    for (int i=0; i < n_cpu; i++) {
        ioctl(fds[i][0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }
    
    // read counters and pack into PyList
    for (int i=0; i < n_cpu; i++) {
        read(fds[i][0], bufs[i], 4096*sizeof(char));
        PyObject* cpu_result = PyList_New(n_event);
        for (int j=0; j < n_event; j++) {
            // search for ids          
            for (int k=0; k < rfs[i]->nr; k++) {
                if (rfs[i]->values[k].id == ids[i][j]) {
                    uint64_t val = rfs[i]->values[k].value;
                    PyList_SetItem(cpu_result,j,Py_BuildValue("l",val));
                }
            }
        }
        PyList_SetItem(result_list,i,cpu_result);
    }

    // free spaces
    for (int i=0; i < n_cpu; i++) {
        for (int j=0; j<n_event;j++){
            close(fds[i][j]);
        }
        free(fds[i]);free(ids[i]);free(bufs[i]);
    }
    free(fds);free(ids);free(bufs);free(rfs);
    return result_list;
}

PyObject*
get_supported_names()
{   
    int length = sizeof(EVENT_LIST)/sizeof(EVENT_LIST[0]);
    PyObject* name_list = PyList_New(length);
    for (int i = 0; i < length; ++i) {
        PyList_SetItem(name_list,i,Py_BuildValue("s",EVENT_LIST[i].name));
    }
    return name_list;    
}

PyObject*
get_supported_abbrevs()
{   
    int length = sizeof(EVENT_LIST)/sizeof(EVENT_LIST[0]);
    PyObject* result_list = PyList_New(length);
    for (int i = 0; i < length; ++i) {
        PyList_SetItem(result_list,i,Py_BuildValue("s",EVENT_LIST[i].abbrev));
    }
    return result_list;    
}

PyObject*
get_supported_events()
{   
    int length = sizeof(EVENT_LIST)/sizeof(EVENT_LIST[0]);
    PyObject* result_list = PyList_New(length);
    for (int i = 0; i < length; ++i) {
        PyList_SetItem(result_list,i,Py_BuildValue("[ssi]",
            EVENT_LIST[i].name,EVENT_LIST[i].abbrev,EVENT_LIST[i].val));
    }
    return result_list;    
}

int
test_param(const int a, PyObject* args)
{   
    // PyObject* list;
    // if (!PyArg_Parse(args, "O!", &PyList_Type, &list)) 
    //     return 0;
    int my_val = (int)PyLong_AsLong(PyList_GetItem(args,0));
    return my_val;

}

