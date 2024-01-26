import numpy as np

from pyatf import TP, Interval, Tuner
from pyatf.cost_functions import python
from pyatf.search_techniques import AUCBandit
from pyatf.tuning_data import Configuration


class TunableGaussian:
    def __init__(self, configuration: Configuration):
        self._cache_block_size = configuration['CACHE_BLOCK_SIZE']

    # computes gaussian in-place in `data` for simplicity
    def __call__(self, data: np.ndarray, N: int):
        for cache_block_offset in range(0, N, self._cache_block_size):
            for i in range(cache_block_offset, cache_block_offset + self._cache_block_size):
                for j in range(N):
                    data[i+1, j+1] = ( data[ i  ,j ] + data[ i  ,j+1 ] + data[ i  ,j+2 ] +
                                       data[ i+1,j ] + data[ i+1,j+1 ] + data[ i+1,j+2 ] +
                                       data[ i+2,j ] + data[ i+2,j+1 ] + data[ i+2,j+2 ] ) / 9.0


# input size
N = 1000

# Step 1: Generate the Search Space
CACHE_BLOCK_SIZE = TP( 'CACHE_BLOCK_SIZE'                                 ,
                       Interval( 1,N )                                    ,
                       lambda CACHE_BLOCK_SIZE: N % CACHE_BLOCK_SIZE == 0 )

# Steps 2 & 3: Program-Guided Search Space Exploration
tuner = Tuner().tuning_parameters( CACHE_BLOCK_SIZE )  \
               .search_technique( AUCBandit() )

data = np.random.rand( N+3,N+3 ).astype(np.float32)

gaussian_cf = python.CostFunction(TunableGaussian)( data, N )

for _ in range(8):
    tuner.make_step( gaussian_cf )
