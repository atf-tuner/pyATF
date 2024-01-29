import numpy as np

from pyatf import TP, Interval, Tuner
from pyatf.cost_functions import cuda
from pyatf.search_techniques import AUCBandit
from pyatf.abort_conditions import Evaluations

# kernel code as string
saxpy_kernel_as_string = '''
extern "C" __global__ void saxpy( const int N, const float a, const float* x, float* y )
{
    for( int w = 0 ; w < WPT ; ++w )
    {
        const int index = w * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        y[ index ] += a * x[ index ];
    }
}
'''

# input size
N = 1000

# Step 1: Generate the Search Space
WPT = TP('WPT', Interval( 1, N ), lambda WPT: N % WPT == 0           )
LS  = TP('LS',  Interval( 1, N ), lambda WPT, LS: (N / WPT) % LS == 0)

# Step 2: Implement a Cost Function
saxpy_kernel = cuda.Kernel( cuda.source(saxpy_kernel_as_string), 'saxpy' )  # kernel's code & name

cf_saxpy = cuda.CostFunction( saxpy_kernel ).device_id( 0 )                                  \
                                            .inputs( np.int32( N )                        ,
                                                     np.float32(np.random.random())       ,
                                                     np.random.rand(N).astype(np.float32) ,
                                                     np.random.rand(N).astype(np.float32) )  \
                                            .grid_dim( lambda WPT, LS: N/WPT/LS )            \
                                            .block_dim( lambda LS: LS )

# Step 3: Explore the Search Space
tuning_result = Tuner().tuning_parameters( WPT, LS )       \
                       .search_technique( AUCBandit() )    \
                       .tune( cf_saxpy, Evaluations(50) )
