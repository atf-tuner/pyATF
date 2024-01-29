from datetime import timedelta

from pyatf import TP, Interval, Set, Tuner
from pyatf.cost_functions import generic
from pyatf.search_techniques import AUCBandit
from pyatf.abort_conditions import Duration

# Step 1: Generate the Search Space
opt_level            = TP('opt_level',            Set( '-O0', '-O1', '-O2', '-O3' )                  )
align_functions      = TP('align_functions',      Set( '-falign-functions', '-fno-align-functions' ) )
early_inlining_insns = TP('early_inlining_insns', Interval( 0, 1000 )                                )

# Step 2: Implement a Cost Function
run_command     = './raytracer/raytracer'
compile_command = './raytracer/compile_raytracer.sh'

generic_cf = generic.CostFunction(run_command).compile_command(compile_command)

# Step 3: Explore the Search Space
tuning_result = Tuner().tuning_parameters( opt_level, align_functions, early_inlining_insns )  \
                       .search_technique( AUCBandit() )                                        \
                       .tune( generic_cf, Duration(timedelta(minutes=5)) )
