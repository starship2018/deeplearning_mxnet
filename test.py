import mxnet as mx
import time

time_start_gpu=time.time()
a = mx.nd.ones((2, 3), mx.gpu())
b = a*2+1

print(b.asnumpy())
time_end_gpu=time.time()

print('gpu_cost', time_end_gpu-time_start_gpu)

time_start_cpu=time.time()
c = mx.nd.ones((2, 3))
d = c*2+1

print(d.asnumpy())
time_end_cpu=time.time()

print('cpu_cost', time_end_cpu-time_start_cpu)