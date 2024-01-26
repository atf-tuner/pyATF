import numpy as np

from pyatf import TP, Interval, Tuner
from pyatf.cost_functions import opencl
from pyatf.result_check import absolute_difference
from pyatf.search_techniques import AUCBandit
from pyatf.abort_conditions import Evaluations

# kernel code as string
sum_kernel_as_string = '''
void atomic_add_f(volatile global float* addr, const float val) {
    private float old, sum;
    do {
        old = *addr;
        sum = old+val;
    } while(atomic_cmpxchg((volatile global int*)addr, as_int(old), as_int(sum))!=as_int(old));
}

__kernel void sum( const int N, const __global float* x, __global float* y )
{
    for( int w = 0 ; w < WPT ; ++w )
    {
        const int index = w * get_global_size(0) + get_global_id(0);
        atomic_add_f( y , x[ index ] );
    }
}
'''

# input size
N = 1000

# compute gold
x = np.random.rand(N).astype(np.float32)
y = np.zeros((1,), dtype=np.float32)
y_gold = np.full((1,), sum(x), dtype=np.float32)

# Step 1: Generate the Search Space
WPT = TP('WPT', Interval( 1, N ), lambda WPT: N % WPT == 0           )
LS  = TP('LS',  Interval( 1, N ), lambda WPT, LS: (N / WPT) % LS == 0)

# Step 2: Implement a Cost Function
sum_kernel = opencl.Kernel( opencl.source(sum_kernel_as_string), 'sum' )  # kernel's code & name

cf_sum = opencl.CostFunction( sum_kernel ).platform_id( 0 )                                                           \
                                          .device_id( 0 )                                                             \
                                          .inputs( np.int32( N ) ,
                                                   x             ,
                                                   y             )                                                    \
                                          .global_size( lambda WPT, LS: N/WPT )                                       \
                                          .local_size( lambda LS: LS )                                                \
                                          .check_result( 2, y_gold,
                                                         absolute_difference(0.001) )                                 \
                                          .check_result( 2, lambda N, x, y: np.full((1,), sum(x), dtype=np.float32),
                                                         absolute_difference(0.001) )

# Step 3: Explore the Search Space
tuning_result = Tuner().tuning_parameters( WPT, LS )       \
                       .search_technique( AUCBandit() )    \
                       .tune( cf_sum, Evaluations(50) )
