import inspect
import time
from functools import reduce
from math import ceil, floor
from typing import List, Optional, Any, SupportsIndex, Dict, Set, Tuple, Union

from pyatf.range import Range
from pyatf.tp import TP
from pyatf.tuning_data import Configuration, Index, Coordinates


class Node:
    def __init__(self, data: Any):
        self._children: List[Node] = []
        self.data = data
        self.num_children: int = 0
        self.num_leafs: int = 0

    def __len__(self):
        return self.num_children

    def __iter__(self):
        yield from self._children

    def add_child(self, child: 'Node'):
        self._children.append(child)
        self.num_children += 1

    def get_child(self, index: SupportsIndex):
        return self._children[index]


class ChainedTree:
    def __init__(self, root_data: Optional[Any] = None):
        self._root = Node(root_data)

    @property
    def root(self):
        return self._root


ChainOfTrees = List[ChainedTree]


class SearchSpace:
    def __init__(self, *tps: TP, enable_1d_access: bool = False, silent: bool = True):
        self._1d_access_enabled: bool = enable_1d_access

        # make TPs accessible by name
        tp_by_name: Dict[str, TP] = {}
        for tp in tps:
            if tp.name in tp_by_name:
                raise ValueError(f'duplicate parameter name: {tp.name}')
            tp_by_name[tp.name] = tp
        self._tp_names: Tuple[str] = tuple(tp_by_name.keys())
        self._num_tps: int = len(tp_by_name)

        # determine independent parameter groups by calculating undirected transitive closure of TPs
        # using Floyd Warshall algorithm on the TPs represented as a reachability matrix
        UNREACHABLE = 'unreachable'
        REFERENCING = 'referencing'
        REFERENCED_BY = 'referenced_by'
        reach = {source_tp: {target_tp: UNREACHABLE for target_tp in tps} for source_tp in tps}
        defined_tps = set()
        for tp in tps:
            defined_tps.add(tp.name)
            if tp.constraint:
                ref_tp_names = set(inspect.signature(tp.constraint).parameters.keys())
                if tp.name not in ref_tp_names:
                    raise ValueError(f'constraint for TP {tp.name} has to take at least itself as a parameter')
                if not ref_tp_names.issubset(defined_tps):
                    raise ValueError(f'constraint for TP {tp.name} references TPs that have not yet been defined: '
                                     f'{", ".join(ref_tp_names.difference(defined_tps))}')
                ref_tp_names.remove(tp.name)
                ref_tps = set(tp_by_name[ref_tp_name] for ref_tp_name in ref_tp_names)
                for ref_tp in ref_tps:
                    reach[tp][ref_tp] = REFERENCING
                    reach[ref_tp][tp] = REFERENCED_BY
        for k in tps:
            for i in tps:
                for j in tps:
                    transitive_reach = None
                    if reach[i][k] != UNREACHABLE and reach[k][j] != UNREACHABLE and reach[i][k] == reach[k][j]:
                        transitive_reach = reach[i][k]
                    if transitive_reach is not None:
                        if reach[i][j] == UNREACHABLE:
                            reach[i][j] = transitive_reach
                        elif reach[i][j] != transitive_reach:
                            raise ValueError(f'circular constraint for TPs {i.name} and {j.name}')
        ungrouped_tps: List[TP] = list(tps)
        tps_to_idx: Dict[TP, int] = {tp: idx for idx, tp in enumerate(tps)}
        independent_tp_groups: List[List[Union[TP, None]]] = []
        while ungrouped_tps:
            tp_group = [None] * self._num_tps
            frontier = [ungrouped_tps[0]]
            while frontier:
                tp = frontier[0]
                del frontier[0]
                tp_group[tps_to_idx[tp]] = tp
                ungrouped_tps.remove(tp)
                for dependent_tp in ungrouped_tps:
                    if reach[tp][dependent_tp] != UNREACHABLE and dependent_tp not in frontier:
                        frontier.append(dependent_tp)
            tp_group = [tp for tp in tp_group if tp is not None]
            independent_tp_groups.append(tp_group)

        # TODO: reorder TPs to allow constraints that reference TPs defined later than the one it is assigned to
        # TODO: optimize TP order to check constraints as early as possible
        # TODO: both operations above should bring TPs into an order independent from the order in which the TPs are defined

        # prepare progress printer
        generation_start_ns = time.perf_counter_ns()
        last_progress_print_ns: Optional[int] = None

        def progress_printer(progress):
            nonlocal last_progress_print_ns
            now = time.perf_counter_ns()
            if last_progress_print_ns is None or now - last_progress_print_ns >= 1000000000:
                if last_progress_print_ns is None:
                    last_progress_print_ns = now
                elapsed_ns = now - generation_start_ns
                elapsed_seconds = elapsed_ns // 1000000000
                elapsed_time_str = (f'{elapsed_seconds // 3600}'
                                    f':{elapsed_seconds // 60 % 60:02d}'
                                    f':{elapsed_seconds % 60:02d}')
                if now > generation_start_ns and progress > 0:
                    eta_seconds = ceil(((now - generation_start_ns) / progress
                                        * (1 - progress)) / 1000000000)
                    eta_str = (f'{eta_seconds // 3600}'
                               f':{eta_seconds // 60 % 60:02d}'
                               f':{eta_seconds % 60:02d}')
                else:
                    eta_str = '?'
                filled = '█' * floor(progress * 79)
                empty = ' ' * ceil((1 - progress) * 79)
                line = (f'\rgenerating search space: |{filled}{empty}|'
                        f' {progress * 100:6.2f}% {elapsed_time_str} (ETA: {eta_str})')
                print(line, end='')

        # generate chain of trees: one tree per independent parameter group
        # TODO: replace children with single node containing TP range,
        #  if the same child exists for each range value (equality has to be checked recursively)
        self._cot: ChainOfTrees = []
        self._cot_layer_to_tp_name: List[str] = []
        self._constrained_size: int = 1 if independent_tp_groups else 0
        self._unconstrained_size: int = 1 if independent_tp_groups else 0
        for tp_group in independent_tp_groups:
            for tp in tp_group:
                self._cot_layer_to_tp_name.append(tp.name)
                self._unconstrained_size *= len(tp.values)
        self._partial_leaf_configs: List[List[Tuple[any, ...]]] = []
        self._num_leafs: List[int] = []
        total_iterations = 0
        tp_to_range_size: List[List[int]] = []
        finished_iterations = 0
        if enable_1d_access:
            if not silent:
                # enable 1D access, with progress prints
                for tp_group in independent_tp_groups:
                    tp_to_range_size.append([])
                    num_leafs = 1
                    for tp in tp_group:
                        tp_to_range_size[-1].append(len(tp.values))
                        num_leafs *= tp_to_range_size[-1][-1]
                    total_iterations += num_leafs
                for tp_group_idx, tp_group in enumerate(independent_tp_groups):
                    self._partial_leaf_configs.append([])
                    if len(tp_group) == 1 and tp_group[0].constraint is None:
                        # for TP groups with a single TP without constraints, conserve storage by storing
                        # the TP range in a single child node instead of generating one node per range value.
                        num_leafs = len(tp_group[0].values)
                        tree = ChainedTree()
                        tree.root.add_child(Node(tp_group[0].values))
                        tree.root.get_child(0).num_leafs = 1
                        finished_iterations += num_leafs
                    else:
                        num_leafs = 0

                        constraint_args: Dict[str, Any] = {tp.name: None for tp in tp_group}
                        tp_to_parameter_names: Dict[TP, Set[str]] = {
                            tp: set(inspect.signature(tp.constraint).parameters.keys()) if tp.constraint else set()
                            for tp in tp_group
                        }

                        def tp_iter(tp_idx: int, tp: TP, constraint_params: Set[str], next_tps: List[TP],
                                    parent_node: Node):
                            nonlocal num_leafs, finished_iterations
                            num_skipped_children = 0
                            for val in tp.values:
                                # check constraint
                                constraint_args[tp.name] = val
                                if tp.constraint and not tp.constraint(**{
                                    param_name: constraint_args[param_name]
                                    for param_name in constraint_params
                                }):
                                    num_skipped_children += 1
                                    continue

                                # create child node, including grandchildren
                                child_node = Node(val)
                                if next_tps:
                                    tp_iter(tp_idx + 1, next_tps[0], tp_to_parameter_names[next_tps[0]], next_tps[1:],
                                            child_node)
                                    # only add child, if it has grandchildren
                                    num_grandchildren = len(child_node)
                                    if num_grandchildren > 0:
                                        parent_node.add_child(child_node)
                                        child_node.num_leafs = 0
                                        for grandchild in child_node:
                                            child_node.num_leafs += grandchild.num_leafs
                                else:
                                    parent_node.add_child(child_node)
                                    child_node.num_leafs = 1
                                    num_leafs += 1
                                    self._partial_leaf_configs[-1].append(tuple(constraint_args.values()))

                            if next_tps:
                                if num_skipped_children > 0:
                                    finished_iterations += num_skipped_children * reduce(
                                        lambda x, y: x * y, tp_to_range_size[tp_group_idx][tp_idx + 1:], 1
                                    )
                            else:
                                finished_iterations += len(tp.values)
                            progress_printer(finished_iterations / total_iterations)

                        tree = ChainedTree()
                        tp_iter(0, tp_group[0], tp_to_parameter_names[tp_group[0]], tp_group[1:], tree.root)
                    tree.root.num_leafs = num_leafs
                    last_progress_print_ns = None  # force print
                    progress_printer(finished_iterations / total_iterations)
                    self._cot.append(tree)
                    self._constrained_size *= num_leafs
                    self._num_leafs.append(num_leafs)
                print('\n')
            else:
                # enable 1D access, no progress prints
                for tp_group_idx, tp_group in enumerate(independent_tp_groups):
                    self._partial_leaf_configs.append([])

                    if len(tp_group) == 1 and tp_group[0].constraint is None:
                        # for TP groups with a single TP without constraints, conserve storage by storing
                        # the TP range in a single child node instead of generating one node per range value.
                        num_leafs = len(tp_group[0].values)
                        tree = ChainedTree()
                        tree.root.add_child(Node(tp_group[0].values))
                        tree.root.get_child(0).num_leafs = 1
                    else:
                        num_leafs = 0

                        constraint_args: Dict[str, Any] = {tp.name: None for tp in tp_group}
                        tp_to_parameter_names: Dict[TP, Set[str]] = {
                            tp: set(inspect.signature(tp.constraint).parameters.keys()) if tp.constraint else set()
                            for tp in tp_group
                        }

                        def tp_iter(tp: TP, constraint_params: Set[str], next_tps: List[TP], parent_node: Node):
                            nonlocal num_leafs
                            for val in tp.values:
                                # check constraint
                                constraint_args[tp.name] = val
                                if tp.constraint and not tp.constraint(**{
                                    param_name: constraint_args[param_name]
                                    for param_name in constraint_params
                                }):
                                    continue

                                # create child node, including grandchildren
                                child_node = Node(val)
                                if next_tps:
                                    tp_iter(next_tps[0], tp_to_parameter_names[next_tps[0]], next_tps[1:], child_node)
                                    # only add child, if it has grandchildren
                                    num_grandchildren = len(child_node)
                                    if num_grandchildren > 0:
                                        parent_node.add_child(child_node)
                                        child_node.num_leafs = 0
                                        for grandchild in child_node:
                                            child_node.num_leafs += grandchild.num_leafs
                                else:
                                    parent_node.add_child(child_node)
                                    child_node.num_leafs = 1
                                    num_leafs += 1
                                    self._partial_leaf_configs[-1].append(tuple(constraint_args.values()))

                        tree = ChainedTree()
                        tp_iter(tp_group[0], tp_to_parameter_names[tp_group[0]], tp_group[1:], tree.root)
                    tree.root.num_leafs = num_leafs
                    self._cot.append(tree)
                    self._constrained_size *= num_leafs
                    self._num_leafs.append(num_leafs)
        else:
            if not silent:
                # no 1D access, with progress prints
                for tp_group in independent_tp_groups:
                    tp_to_range_size.append([])
                    num_leafs = 1
                    for tp in tp_group:
                        tp_to_range_size[-1].append(len(tp.values))
                        num_leafs *= tp_to_range_size[-1][-1]
                    total_iterations += num_leafs
                for tp_group_idx, tp_group in enumerate(independent_tp_groups):
                    if len(tp_group) == 1 and tp_group[0].constraint is None:
                        # for TP groups with a single TP without constraints, conserve storage by storing
                        # the TP range in a single child node instead of generating one node per range value.
                        num_leafs = len(tp_group[0].values)
                        tree = ChainedTree()
                        tree.root.add_child(Node(tp_group[0].values))
                        tree.root.get_child(0).num_leafs = 1
                        finished_iterations += num_leafs
                    else:
                        num_leafs = 0

                        constraint_args: Dict[str, Any] = {tp.name: None for tp in tp_group}
                        tp_to_parameter_names: Dict[TP, Set[str]] = {
                            tp: set(inspect.signature(tp.constraint).parameters.keys()) if tp.constraint else set()
                            for tp in tp_group
                        }

                        def tp_iter(tp_idx: int, tp: TP, constraint_params: Set[str], next_tps: List[TP],
                                    parent_node: Node):
                            nonlocal num_leafs, finished_iterations
                            num_skipped_children = 0
                            for val in tp.values:
                                # check constraint
                                constraint_args[tp.name] = val
                                if tp.constraint and not tp.constraint(**{
                                    param_name: constraint_args[param_name]
                                    for param_name in constraint_params
                                }):
                                    num_skipped_children += 1
                                    continue
                                # create child node, including grandchildren
                                child_node = Node(val)
                                if next_tps:
                                    tp_iter(tp_idx + 1, next_tps[0], tp_to_parameter_names[next_tps[0]], next_tps[1:],
                                            child_node)
                                    # only add child, if it has grandchildren
                                    num_grandchildren = len(child_node)
                                    if num_grandchildren > 0:
                                        parent_node.add_child(child_node)
                                        child_node.num_leafs = 0
                                        for grandchild in child_node:
                                            child_node.num_leafs += grandchild.num_leafs
                                else:
                                    parent_node.add_child(child_node)
                                    child_node.num_leafs = 1
                                    num_leafs += 1
                            if next_tps:
                                if num_skipped_children > 0:
                                    finished_iterations += num_skipped_children * reduce(
                                        lambda x, y: x * y, tp_to_range_size[tp_group_idx][tp_idx + 1:], 1
                                    )
                            else:
                                finished_iterations += len(tp.values)
                            progress_printer(finished_iterations / total_iterations)

                        tree = ChainedTree()
                        tp_iter(0, tp_group[0], tp_to_parameter_names[tp_group[0]], tp_group[1:], tree.root)
                    tree.root.num_leafs = num_leafs
                    last_progress_print_ns = None  # force print
                    progress_printer(finished_iterations / total_iterations)
                    self._cot.append(tree)
                    self._constrained_size *= num_leafs
                print('\n')
            else:
                # no 1D access, no progress prints
                for tp_group_idx, tp_group in enumerate(independent_tp_groups):
                    if len(tp_group) == 1 and tp_group[0].constraint is None:
                        # for TP groups with a single TP without constraints, conserve storage by storing
                        # the TP range in a single child node instead of generating one node per range value.
                        num_leafs = len(tp_group[0].values)
                        tree = ChainedTree()
                        tree.root.add_child(Node(tp_group[0].values))
                        tree.root.get_child(0).num_leafs = 1
                    else:
                        num_leafs = 0

                        constraint_args: Dict[str, Any] = {tp.name: None for tp in tp_group}
                        tp_to_parameter_names: Dict[TP, Set[str]] = {
                            tp: set(inspect.signature(tp.constraint).parameters.keys()) if tp.constraint else set()
                            for tp in tp_group
                        }

                        def tp_iter(tp: TP, constraint_params: Set[str], next_tps: List[TP], parent_node: Node):
                            nonlocal num_leafs
                            for val in tp.values:
                                # check constraint
                                constraint_args[tp.name] = val
                                if tp.constraint and not tp.constraint(**{
                                    param_name: constraint_args[param_name]
                                    for param_name in constraint_params
                                }):
                                    continue
                                # create child node, including grandchildren
                                child_node = Node(val)
                                if next_tps:
                                    tp_iter(next_tps[0], tp_to_parameter_names[next_tps[0]], next_tps[1:], child_node)
                                    # only add child, if it has grandchildren
                                    num_grandchildren = len(child_node)
                                    if num_grandchildren > 0:
                                        parent_node.add_child(child_node)
                                        child_node.num_leafs = 0
                                        for grandchild in child_node:
                                            child_node.num_leafs += grandchild.num_leafs
                                else:
                                    parent_node.add_child(child_node)
                                    child_node.num_leafs = 1
                                    num_leafs += 1

                        tree = ChainedTree()
                        tp_iter(tp_group[0], tp_to_parameter_names[tp_group[0]], tp_group[1:], tree.root)
                    tree.root.num_leafs = num_leafs
                    self._cot.append(tree)
                    self._constrained_size *= num_leafs

    def __len__(self):
        return self._constrained_size

    @property
    def cot(self):
        return self._cot

    @property
    def tp_names(self):
        return self._tp_names

    @property
    def num_tps(self):
        return self._num_tps

    @property
    def constrained_size(self):
        return self._constrained_size

    @property
    def unconstrained_size(self):
        return self._unconstrained_size

    def get_configuration(self, coordinates_or_index: Union[Coordinates, Index]) -> Dict[str, Any]:
        if isinstance(coordinates_or_index, Index):
            index = coordinates_or_index
            if index < 0 or index >= self._constrained_size:
                raise ValueError(f'expecting index in range [0,{self._constrained_size})')
            if not self._1d_access_enabled:
                raise ValueError('search space can only be accessed using indices, '
                                 'if it has been generated with enable_1d_access set to True')

            # convert index to one index per tree
            index_per_tree: List[int] = []
            divisor = 1
            for num_leafs in reversed(self._num_leafs):
                index_per_tree.insert(0, (index // divisor) % num_leafs)
                divisor *= num_leafs

            # create configuration from index per tree
            config: Configuration = {}
            layer: int = 0
            trees = iter(self._cot)
            node = next(trees).root
            for leaf_index, partial_leaf_configs in zip(index_per_tree, self._partial_leaf_configs):
                if node.num_children == 0:
                    # go to next tree, if at leaf
                    node = next(trees).root
                if node.num_children == 1 and isinstance(node.get_child(0).data, Range):
                    # TP values are represented by TP range
                    node = node.get_child(0)
                    config[self._cot_layer_to_tp_name[layer]] = node.data[leaf_index]
                    layer += 1
                else:
                    # TP values are represented as child nodes
                    for leaf_value in partial_leaf_configs[leaf_index]:
                        config[self._cot_layer_to_tp_name[layer]] = leaf_value
                        layer += 1

            return config
        else:
            coordinates = coordinates_or_index
            if self._constrained_size == 0:
                raise ValueError('search space does not contain any configurations')
            if len(coordinates) != self._num_tps:
                raise ValueError(f'expecting {self._num_tps} coordinates')
            if any(map(lambda c: c <= 0 or c > 1, coordinates)):
                raise ValueError(f'expecting all coordinates to be in (0,1]:\n{chr(10).join(map(str, coordinates))}')

            config: Configuration = {}
            trees = iter(self._cot)
            node = next(trees).root
            for layer, coordinate in enumerate(coordinates):
                if node.num_children == 0:
                    # go to next tree, if at leaf
                    node = next(trees).root
                if node.num_children == 1 and isinstance(node.get_child(0).data, Range):
                    # TP values are represented by TP range
                    node = node.get_child(0)
                    assert node.num_children == 0  # TPs stored by range are currently only supported for leaf nodes
                    # no bias correction necessary, since node is a leaf
                    index = ceil(coordinate * len(node.data)) - 1
                    config[self._cot_layer_to_tp_name[layer]] = node.data[index]
                else:
                    # TP values are represented as child nodes
                    num_left_leafs = 0
                    for child in node:
                        if num_left_leafs < coordinate * node.num_leafs <= num_left_leafs + child.num_leafs:
                            node = child
                            break
                        else:
                            num_left_leafs += child.num_leafs
                    config[self._cot_layer_to_tp_name[layer]] = node.data

            return config
