import numpy as np

from pyatf import TP, Interval, Tuner, Set
from pyatf.cost_functions import opencl
from pyatf.search_techniques import AUCBandit
from pyatf.abort_conditions import Evaluations

# kernel code from path
sgemm_kernel_as_string = opencl.path('./cltune_gemm.cl')

# input size
M = 8
N = 8
K = 8

# Query Device-Specific OpenCL Limits from pyATF
max_wi_sizes = opencl.max_work_item_sizes()
max_wg_size = opencl.max_work_group_size()
local_mem_size = opencl.local_mem_size()

# Step 1: Generate the Search Space
MWG = TP('MWG', Interval(1, M), lambda MWG: M % MWG == 0)
NWG = TP('NWG', Interval(1, N), lambda NWG: N % NWG == 0)
KWG = TP('KWG', Interval(1, K), lambda KWG: K % KWG == 0)

MDIMC = TP('MDIMC', Interval(1, M), lambda MDIMC, MWG: MWG % MDIMC == 0
                                                    and MDIMC <= max_wi_sizes[0])
NDIMC = TP('NDIMC', Interval(1, N), lambda NDIMC, NWG, MDIMC: NWG % NDIMC == 0
                                                           and NDIMC <= max_wi_sizes[1]
                                                           and MDIMC * NDIMC <= max_wg_size)
MDIMA = TP('MDIMA', Interval(1, M), lambda MDIMA, MWG, NDIMC, KWG, MDIMC: MWG % MDIMA == 0
                                                                       and (MDIMC * NDIMC) % MDIMA == 0
                                                                       and KWG % ((MDIMC * NDIMC) / MDIMA) == 0)
NDIMB = TP('NDIMB', Interval(1, N), lambda NDIMB, NWG, NDIMC, KWG, MDIMC: NWG % NDIMB == 0
                                                                       and (MDIMC * NDIMC) % NDIMB == 0
                                                                       and KWG % ((MDIMC * NDIMC) / NDIMB) == 0)

KWI = TP('KWI', Interval(1, K), lambda KWI, KWG: KWG % KWI == 0)

VWM = TP('VWM', Set(1, 2, 4, 8), lambda VWM, MWG, MDIMC, MDIMA: (MWG / MDIMC) % VWM == 0
                                                                and (MWG / MDIMA) % VWM == 0)
VWN = TP('VWN', Set(1, 2, 4, 8), lambda VWN, NWG, NDIMC, NDIMB: (NWG / NDIMC) % VWN == 0
                                                                and (NWG / NDIMB) % VWN == 0)

STRM = TP('STRM', Set(0, 1))
STRN = TP('STRN', Set(0, 1))

SA = TP('SA', Set(0, 1))
SB = TP('SB', Set(0, 1),
        lambda SB, SA, KWG, MWG, VWM, NWG, VWN: ((SA * KWG * MWG / VWM) + (SB * KWG * NWG / VWN)) * 8 <= local_mem_size
        )  # restriction of local memory

# Step 2: Implement a Cost Function
saxpy_kernel = opencl.Kernel( opencl.source(sgemm_kernel_as_string), 'gemm_fast', ['-DPRECISION=32'] )  # kernel's code & name

cf_saxpy = opencl.CostFunction( saxpy_kernel ).platform_id( 0 )                                   \
                                              .device_id( 0 )                                     \
                                              .inputs( np.int32( M )                            ,
                                                       np.int32( N )                            ,
                                                       np.int32( K )                            ,
                                                       np.random.rand( M*K ).astype(np.float32) ,
                                                       np.random.rand( N*K ).astype(np.float32) ,
                                                       np.random.rand( M*N ).astype(np.float32) ) \
                                              .global_size( lambda MWG, MDIMC: ((1 + ((M - 1) // MWG))*MWG * MDIMC) / MWG,
                                                            lambda NWG, NDIMC: ((1 + ((N - 1) // NWG))*NWG * NDIMC) / NWG) \
                                              .local_size( lambda MDIMC: MDIMC,
                                                           lambda NDIMC: NDIMC )

# Step 3: Explore the Search Space
tuning_result = Tuner().tuning_parameters( MWG,NWG,KWG,
                                           MDIMC,NDIMC,MDIMA,NDIMB,
                                           KWI,
                                           VWM,VWN,
                                           STRM,STRN,
                                           SA,SB )                   \
                       .search_technique( AUCBandit() )              \
                       .tune( cf_saxpy, Evaluations(50) )
