# coding: utf-8

# # Technoedge Quickstart

# In[158]:


import ctypes
from time import time

import numpy as np
import pynq
from pynq import Overlay

# ## Wrapper Class

# In[169]:


STATUS_REGISTER = 0x0

DATA_ADDR = 512
RESULT_ADDR = 74256

FC1_WT_ADDR = 32768
FC1_BIAS_ADDR = 65536

FC2_WT_ADDR = 69632
FC2_BIAS_ADDR = 73728

FC3_WT_ADDR = 73984
FC3_BIAS_ADDR = 74240

FC1_IN = 84
FC1_OUT = 64
FC2_OUT = 16
FC3_OUT = 3

FIXED_FACTOR = 2 ** 24


def to_int_arr(arr):
    return (
        np.array([xi if xi >= 0 else xi + 2 ** 32 for xi in arr]).astype(int).tolist()
    )


class TechnoEdge:
    def __init__(self, filename):
        self.overlay = Overlay(filename)
        self.techno = self.overlay.technoedge_0

    def write(self, data):
        arr = self.techno.mmio.array[DATA_ADDR // 4 : DATA_ADDR // 4 + FC1_IN]
        arr[:] = to_int_arr(data)
        # for i in range(FC1_IN):
        #    self.techno.write(DATA_ADDR + i * 4, int(data[i]))

    def run(self):
        self.techno.write(STATUS_REGISTER, 1)
        while self.techno.read(STATUS_REGISTER) & 0x2 == 0:
            pass

    def run_benchmark(self, reps=1000):
        start = time()
        for _ in range(reps):
            self.write([0] * FC1_IN)
            self.run()
            ignore = self.get_result()
        end = time()

        return (end - start) / reps * 1000000 / 3

    def get_result(self):
        result = []
        for i in range(FC3_OUT):
            val = self.techno.read(RESULT_ADDR + i * 4)
            val = ctypes.c_int(val).value
            result.append(val / FIXED_FACTOR)

        return result

    def put_weights(self, wts):
        #
        # FC1 weight and bias
        #
        arr = self.techno.mmio.array[
            FC1_WT_ADDR // 4 : FC1_WT_ADDR // 4 + len(wts["fc1_wt"])
        ]
        arr[:] = to_int_arr(wts["fc1_wt"])
        arr = self.techno.mmio.array[
            FC1_BIAS_ADDR // 4 : FC1_BIAS_ADDR // 4 + len(wts["fc1_bias"])
        ]
        arr[:] = to_int_arr(wts["fc1_bias"])

        #
        # FC2 weight and bias
        #
        arr = self.techno.mmio.array[
            FC2_WT_ADDR // 4 : FC2_WT_ADDR // 4 + len(wts["fc2_wt"])
        ]
        arr[:] = to_int_arr(wts["fc2_wt"])
        arr = self.techno.mmio.array[
            FC2_BIAS_ADDR // 4 : FC2_BIAS_ADDR // 4 + len(wts["fc2_bias"])
        ]
        arr[:] = to_int_arr(wts["fc2_bias"])

        #
        # FC3 weight and bias
        #
        arr = self.techno.mmio.array[
            FC3_WT_ADDR // 4 : FC3_WT_ADDR // 4 + len(wts["fc3_wt"])
        ]
        arr[:] = to_int_arr(wts["fc3_wt"])
        arr = self.techno.mmio.array[
            FC3_BIAS_ADDR // 4 : FC3_BIAS_ADDR // 4 + len(wts["fc3_bias"])
        ]
        arr[:] = to_int_arr(wts["fc3_bias"])
