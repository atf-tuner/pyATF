import inspect
import time
from math import ceil
from typing import Union, Tuple, Callable, Optional, Set, Dict, Iterable

import numpy
import pyopencl as cl

from pyatf.result_check import equality
from pyatf.tuning_data import Configuration, Cost, MetaData, CostFunctionError


def get_device(platform_id: int = 0, device_id: int = 0) -> cl.Device:
    platforms = cl.get_platforms()
    if platform_id >= len(platforms):
        raise ValueError(f'invalid platform id: {platform_id}')
    cl_platform = platforms[platform_id]
    devices = cl_platform.get_devices()
    if device_id >= len(devices):
        raise ValueError(f'invalid device id: {device_id}')
    return devices[device_id]


def local_mem_size(platform_id: int = 0, device_id: int = 0):
    device = get_device(platform_id, device_id)
    return device.get_info(cl.device_info.LOCAL_MEM_SIZE)


def max_work_item_sizes(platform_id: int = 0, device_id: int = 0):
    device = get_device(platform_id, device_id)
    return device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)


def max_work_group_size(platform_id: int = 0, device_id: int = 0):
    device = get_device(platform_id, device_id)
    return device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)


def source(source: str):
    return source


def path(path: str):
    with open(path, 'r') as source:
        return source.read()


class Kernel:
    def __init__(self, source: str, name: str = 'func', flags: Iterable[str] = None):
        self._source = source
        self._name = name
        if flags is not None:
            self._flags = tuple(flags)
        else:
            self._flags = tuple()

    @property
    def source(self):
        return self._source

    @property
    def name(self):
        return self._name

    @property
    def flags(self):
        return self._flags


Input = Union[numpy.ndarray, numpy.generic]


class CostFunction:
    def __init__(self, kernel: Kernel):
        self._kernel = kernel

        self._silent: bool = False
        self._platform_id: Optional[int] = None
        self._device_id: Optional[int] = None

        self._inputs: Dict[int, Input] = {}

        self._global_size_x: Union[int, Callable[..., int]] = 1
        self._global_size_x_tps: Optional[Set[str, ...]] = None
        self._global_size_y: Union[int, Callable[..., int]] = 1
        self._global_size_y_tps: Optional[Set[str, ...]] = None
        self._global_size_z: Union[int, Callable[..., int]] = 1
        self._global_size_z_tps: Optional[Set[str, ...]] = None
        self._local_size_x: Union[int, Callable[..., int]] = 1
        self._local_size_x_tps: Optional[Set[str, ...]] = None
        self._local_size_y: Union[int, Callable[..., int]] = 1
        self._local_size_y_tps: Optional[Set[str, ...]] = None
        self._local_size_z: Union[int, Callable[..., int]] = 1
        self._local_size_z_tps: Optional[Set[str, ...]] = None

        self._gold_data: Dict[int, Tuple[Input, Callable[[numpy.generic, numpy.generic], bool]]] = {}
        self._gold_cmp_buffer: Dict[int, Input] = {}

        self._warmups: int = 0
        self._evaluations: int = 1

        self._cl_platform: Optional[cl.Platform] = None
        self._cl_device: Optional[cl.Device] = None
        self._cl_context: Optional[cl.Context] = None
        self._cl_queue: Optional[cl.CommandQueue] = None
        self._cl_buffer: Dict[int, cl.Buffer] = {}

    def __del__(self):
        self._free_buffers()
        self._free_command_queue()

    def silent(self, silent: bool):
        self._silent = silent

    def platform_id(self, platform_id: int):
        self._free_buffers()
        self._free_command_queue()
        self._platform_id = platform_id
        self._init_command_queue()
        self._alloc_buffers()
        return self

    def device_id(self, device_id: int):
        self._free_buffers()
        self._free_command_queue()
        self._device_id = device_id
        self._init_command_queue()
        self._alloc_buffers()
        return self

    def inputs(self, *inputs: Input):
        self._free_buffers()
        self._inputs = {idx: inp for idx, inp in enumerate(inputs)}
        self._alloc_buffers()
        return self

    def global_size(self,
                    x: Union[int, Callable[..., int]],
                    y: Union[int, Callable[..., int]] = 1,
                    z: Union[int, Callable[..., int]] = 1):
        self._global_size_x = x
        if type(x) == int:
            self._global_size_x_tps = None
        else:
            self._global_size_x_tps = set(inspect.signature(x).parameters.keys())
        self._global_size_y = y
        if type(y) == int:
            self._global_size_y_tps = None
        else:
            self._global_size_y_tps = set(inspect.signature(y).parameters.keys())
        self._global_size_z = z
        if type(z) == int:
            self._global_size_z_tps = None
        else:
            self._global_size_z_tps = set(inspect.signature(z).parameters.keys())
        return self

    def local_size(self,
                   x: Union[int, Callable[..., int]],
                   y: Union[int, Callable[..., int]] = 1,
                   z: Union[int, Callable[..., int]] = 1):
        self._local_size_x = x
        if type(x) == int:
            self._local_size_x_tps = None
        else:
            self._local_size_x_tps = set(inspect.signature(x).parameters.keys())
        self._local_size_y = y
        if type(y) == int:
            self._local_size_y_tps = None
        else:
            self._local_size_y_tps = set(inspect.signature(y).parameters.keys())
        self._local_size_z = z
        if type(z) == int:
            self._local_size_z_tps = None
        else:
            self._local_size_z_tps = set(inspect.signature(z).parameters.keys())
        return self

    def check_result(self, index: int, gold_data_or_callable: Union[Input, Callable[..., Input]],
                     comparator: Callable[[numpy.generic, numpy.generic], bool] = equality):
        if isinstance(gold_data_or_callable, (numpy.ndarray, numpy.generic)):
            self._gold_data[index] = (gold_data_or_callable, comparator)
            self._gold_cmp_buffer[index] = gold_data_or_callable.copy()
        else:
            gold_buffer = gold_data_or_callable(*self._inputs.values())
            self._gold_data[index] = (gold_buffer, comparator)
            self._gold_cmp_buffer[index] = gold_buffer.copy()
        return self

    def warmups(self, warmups: int):
        self._warmups = warmups

    def evaluations(self, evaluations: int):
        self._evaluations = evaluations

    def __call__(self, configuration: Configuration) -> Tuple[Cost, MetaData]:
        if self._platform_id is None or self._device_id is None:
            raise ValueError('no OpenCL platform or device was selected')

        meta_data = {}

        try:
            # create & build program, and get kernel object
            cl_program = cl.Program(self._cl_context, self._kernel.source)
            opts = list(self._kernel.flags)
            for tp_name, tp_value in configuration.items():
                opts.append(f'-D{tp_name}={tp_value}')
            build_start_ns = time.perf_counter_ns()
            cl_program.build(opts)
            build_end_ns = time.perf_counter_ns()
            meta_data['build_time_ns'] = build_end_ns - build_start_ns
            kernel = getattr(cl_program, self._kernel.name)

            # calculate global and local size
            global_size = [1, 1, 1]
            if self._global_size_x_tps is None:
                global_size[0] = self._global_size_x
            else:
                global_size[0] = int(self._global_size_x(**{
                    tp_name: configuration[tp_name] for tp_name in self._global_size_x_tps
                }))
            if self._global_size_y_tps is None:
                global_size[1] = self._global_size_y
            else:
                global_size[1] = int(self._global_size_y(**{
                    tp_name: configuration[tp_name] for tp_name in self._global_size_y_tps
                }))
            if self._global_size_z_tps is None:
                global_size[2] = self._global_size_z
            else:
                global_size[2] = int(self._global_size_z(**{
                    tp_name: configuration[tp_name] for tp_name in self._global_size_z_tps
                }))
            local_size = [1, 1, 1]
            if self._local_size_x_tps is None:
                local_size[0] = self._local_size_x
            else:
                local_size[0] = int(self._local_size_x(**{
                    tp_name: configuration[tp_name] for tp_name in self._local_size_x_tps
                }))
            if self._local_size_y_tps is None:
                local_size[1] = self._local_size_y
            else:
                local_size[1] = int(self._local_size_y(**{
                    tp_name: configuration[tp_name] for tp_name in self._local_size_y_tps
                }))
            if self._local_size_z_tps is None:
                local_size[2] = self._local_size_z
            else:
                local_size[2] = int(self._local_size_z(**{
                    tp_name: configuration[tp_name] for tp_name in self._local_size_z_tps
                }))

            # set kernel arguments
            for idx, inp in self._inputs.items():
                if isinstance(inp, numpy.ndarray):
                    kernel.set_arg(idx, self._cl_buffer[idx])
                else:
                    kernel.set_arg(idx, inp)

            # warmups
            for _ in range(self._warmups):
                self._cpy_to_device()
                cl.enqueue_nd_range_kernel(self._cl_queue, kernel, global_size, local_size)

            # evaluations
            avg_runtime = 0.0
            for e in range(self._evaluations):
                self._cpy_to_device()
                kernel_event = cl.enqueue_nd_range_kernel(self._cl_queue, kernel, global_size, local_size)
                kernel_event.wait()
                avg_runtime += (kernel_event.get_profiling_info(cl.profiling_info.END)
                                - kernel_event.get_profiling_info(cl.profiling_info.START))
                # result check
                if e == 0 and self._gold_data:
                    self._cpy_to_host(self._gold_cmp_buffer)
                    for idx, (gold_values, comparator) in self._gold_data.items():
                        result_values = self._gold_cmp_buffer[idx]
                        if result_values.size != gold_values.size:
                            meta_data['result_check'] = {
                                'status': 'failed',
                                'input': idx,
                                'reason': 'result size is not equal to gold size'
                            }
                            raise CostFunctionError(meta_data)
                        for value_idx, (result_value, gold_value) in enumerate(zip(result_values, gold_values)):
                            if not comparator(result_value, gold_value):
                                meta_data['result_check'] = {
                                    'status': 'failed',
                                    'input': idx,
                                    'position': value_idx,
                                    'expected': str(gold_value),
                                    'actual': str(result_value)
                                }
                                raise CostFunctionError(meta_data)
                    meta_data['result_check'] = {'status': 'success', 'checked_inputs': tuple(self._gold_data.keys())}
            avg_runtime /= self._evaluations
            avg_runtime = float(ceil(avg_runtime))  # runtime can be ceiled, since nanoseconds are precise enough
        except cl.Error as e:
            meta_data['opencl_error_code'] = e.code
            meta_data['opencl_error_what'] = e.what.what()
            meta_data['opencl_error_routine'] = e.routine
            raise CostFunctionError(meta_data) from e
        finally:
            # free program resources
            del kernel
            del cl_program

        return avg_runtime, meta_data

    def _init_command_queue(self):
        if self._platform_id is not None and self._device_id is not None:
            platforms = cl.get_platforms()
            if self._platform_id >= len(platforms):
                raise ValueError(f'invalid platform id: {self._platform_id}')
            self._cl_platform = platforms[self._platform_id]
            if not self._silent:
                print(f'selecting OpenCL platform {self._platform_id}: {self._cl_platform.get_info(cl.platform_info.NAME)}')
            devices = self._cl_platform.get_devices()
            if self._device_id >= len(devices):
                raise ValueError(f'invalid device id: {self._device_id}')
            self._cl_device = devices[self._device_id]
            if not self._silent:
                print(f'selecting OpenCL device {self._device_id}: {self._cl_device.get_info(cl.device_info.NAME)}')
            self._cl_context = cl.Context(devices=[self._cl_device])
            self._cl_queue = cl.CommandQueue(self._cl_context, self._cl_device,
                                             cl.command_queue_properties.PROFILING_ENABLE)

    def _alloc_buffers(self):
        if self._cl_context is not None:
            self._cl_buffer = {
                idx: cl.Buffer(self._cl_context, cl.mem_flags.READ_WRITE, inp.nbytes)
                for idx, inp in self._inputs.items() if isinstance(inp, numpy.ndarray)
            }

    def _cpy_to_device(self):
        if self._cl_queue is not None:
            for idx, buffer in self._cl_buffer.items():
                ndarray = self._inputs[idx]
                cl.enqueue_copy(self._cl_queue, buffer, ndarray)
            self._cl_queue.finish()

    def _cpy_to_host(self, host_memory: Dict[int, numpy.ndarray] = None):
        if self._cl_queue is not None:
            if host_memory is None:
                host_memory = self._inputs
            for idx, ndarray in host_memory.items():
                if isinstance(ndarray, numpy.ndarray):
                    cl.enqueue_copy(self._cl_queue, ndarray, self._cl_buffer[idx])
            self._cl_queue.finish()

    def _free_buffers(self):
        self._cl_buffer.clear()

    def _free_command_queue(self):
        del self._cl_queue
        self._cl_queue = None
        del self._cl_context
        self._cl_context = None
        del self._cl_device
        self._cl_device = None
        del self._cl_platform
        self._cl_platform = None
