from pyatf import TP, Interval, Tuner
from pyatf.cost_functions import generic
from pyatf.search_techniques import Exhaustive

# Step 1: Generate the Search Space
BLOCK_SIZE = TP('BLOCK_SIZE', Interval(1, 10))

# Step 2: Implement a Cost Function
run_command     = './tmp.bin'
compile_command = 'g++ ../mmm_block.cpp -DBLOCK_SIZE=$BLOCK_SIZE -o ./tmp.bin'

generic_cf = generic.CostFunction(run_command).compile_command(compile_command)

# Step 3: Explore the Search Space
tuning_result = Tuner().tuning_parameters( BLOCK_SIZE )   \
                       .search_technique( Exhaustive() )  \
                       .tune( generic_cf )
