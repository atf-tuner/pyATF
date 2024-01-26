import inspect
import time
from math import ceil
from typing import Union, Tuple, Callable, Optional, Set, Dict, Iterable, Any

import numpy
from cuda import cuda, nvrtc

from pyatf.result_check import equality
from pyatf.tuning_data import Configuration, Cost, MetaData, CostFunctionError

cuda.cuInit(0)


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
        self._device_id: Optional[int] = None

        self._inputs: Dict[int, Input] = {}

        self._grid_dim_x: Union[int, Callable[..., int]] = 1
        self._grid_dim_x_tps: Optional[Set[str, ...]] = None
        self._grid_dim_y: Union[int, Callable[..., int]] = 1
        self._grid_dim_y_tps: Optional[Set[str, ...]] = None
        self._grid_dim_z: Union[int, Callable[..., int]] = 1
        self._grid_dim_z_tps: Optional[Set[str, ...]] = None
        self._block_dim_x: Union[int, Callable[..., int]] = 1
        self._block_dim_x_tps: Optional[Set[str, ...]] = None
        self._block_dim_y: Union[int, Callable[..., int]] = 1
        self._block_dim_y_tps: Optional[Set[str, ...]] = None
        self._block_dim_z: Union[int, Callable[..., int]] = 1
        self._block_dim_z_tps: Optional[Set[str, ...]] = None

        self._gold_data: Dict[int, Tuple[Input, Callable[[numpy.generic, numpy.generic], bool]]] = {}
        self._gold_cmp_buffer: Dict[int, Input] = {}

        self._warmups: int = 0
        self._evaluations: int = 1

        self._objects_to_free_on_error: Dict[Any, Callable] = {}
        self._cu_device: Optional[cuda.CUdevice] = None
        self._cu_context: Optional[cuda.CUcontext] = None
        self._cu_stream: Optional[cuda.CUstream] = None
        self._cu_device_ptr: Dict[int, cuda.CUdeviceptr] = {}

    def _free_objects_before_error(self):
        for obj, free_method in self._objects_to_free_on_error.items():
            free_method(obj)
        self._objects_to_free_on_error.clear()

    def _safe_call(
            self,
            returns: Union[cuda.CUresult, nvrtc.nvrtcResult, Tuple[Union[cuda.CUresult, nvrtc.nvrtcResult], ...]],
            message: str, meta_data: Optional[MetaData] = None
    ):
        if isinstance(returns, tuple):
            err = returns[0]
            returns = returns[1:]
        else:
            err = returns
        raise_error = False
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                if meta_data is not None:
                    meta_data['cuda_error'] = str(err)
                raise_error = True
        elif isinstance(err, nvrtc.nvrtcResult):
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                if meta_data is not None:
                    meta_data['nvrtc_error'] = str(err)
                raise_error = True
        else:
            raise_error = True
        if raise_error:
            self._free_objects_before_error()
            if meta_data is not None:
                raise CostFunctionError(meta_data)
            else:
                raise ValueError(f'CUDA failed with error: {err}\n{message}')
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    def __del__(self):
        self._free_buffers()
        self._free_device()

    def silent(self, silent: bool):
        self._silent = silent

    def device_id(self, device_id: int):
        self._free_buffers()
        self._free_device()
        self._device_id = device_id
        self._init_device()
        self._alloc_buffers()
        return self

    def inputs(self, *inputs: Input):
        self._free_buffers()
        self._inputs = {idx: inp for idx, inp in enumerate(inputs)}
        self._alloc_buffers()
        return self

    def grid_dim(self,
                 x: Union[int, Callable[..., int]],
                 y: Union[int, Callable[..., int]] = 1,
                 z: Union[int, Callable[..., int]] = 1):
        self._grid_dim_x = x
        if type(x) == int:
            self._grid_dim_x_tps = None
        else:
            self._grid_dim_x_tps = set(inspect.signature(x).parameters.keys())
        self._grid_dim_y = y
        if type(y) == int:
            self._grid_dim_y_tps = None
        else:
            self._grid_dim_y_tps = set(inspect.signature(y).parameters.keys())
        self._grid_dim_z = z
        if type(z) == int:
            self._grid_dim_z_tps = None
        else:
            self._grid_dim_z_tps = set(inspect.signature(z).parameters.keys())
        return self

    def block_dim(self,
                  x: Union[int, Callable[..., int]],
                  y: Union[int, Callable[..., int]] = 1,
                  z: Union[int, Callable[..., int]] = 1):
        self._block_dim_x = x
        if type(x) == int:
            self._block_dim_x_tps = None
        else:
            self._block_dim_x_tps = set(inspect.signature(x).parameters.keys())
        self._block_dim_y = y
        if type(y) == int:
            self._block_dim_y_tps = None
        else:
            self._block_dim_y_tps = set(inspect.signature(y).parameters.keys())
        self._block_dim_z = z
        if type(z) == int:
            self._block_dim_z_tps = None
        else:
            self._block_dim_z_tps = set(inspect.signature(z).parameters.keys())
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
        if self._device_id is None:
            raise ValueError('no CUDA device was selected')

        meta_data = {}

        # create & compile program
        nvrtc_program = self._safe_call(nvrtc.nvrtcCreateProgram(str.encode(self._kernel.source),
                                                                 str.encode(self._kernel.name + '.cu'),
                                                                 0, [], []),
                                        'failed to create NVRTC program', meta_data)
        self._objects_to_free_on_error[nvrtc_program] = nvrtc.nvrtcDestroyProgram
        opts = []
        for flag in self._kernel.flags:
            opts.append(str.encode(flag))
        for tp_name, tp_value in configuration.items():
            opts.append(str.encode(f'-D{tp_name}={tp_value}'))
        compile_start_ns = time.perf_counter_ns()
        self._safe_call(nvrtc.nvrtcCompileProgram(nvrtc_program, len(opts), opts), 'failed to compile NVRTC program',
                        meta_data)
        compile_end_ns = time.perf_counter_ns()
        meta_data['compile_time_ns'] = compile_end_ns - compile_start_ns
        ptx_size = self._safe_call(nvrtc.nvrtcGetPTXSize(nvrtc_program), 'failed to get PTX size',
                                   meta_data)
        ptx = b' ' * ptx_size
        self._safe_call(nvrtc.nvrtcGetPTX(nvrtc_program, ptx), 'failed to get PTX code',
                        meta_data)

        # load PTX as module and retrieve function
        ptx = numpy.char.array(ptx)
        module = self._safe_call(cuda.cuModuleLoadData(ptx.ctypes.data), 'failed to load PTX',
                                 meta_data)
        self._objects_to_free_on_error[module] = cuda.cuModuleUnload
        kernel = self._safe_call(cuda.cuModuleGetFunction(module, str.encode(self._kernel.name)),
                                 'failed to get module function', meta_data)

        # calculate grid and block dim
        grid_dim = [1, 1, 1]
        if self._grid_dim_x_tps is None:
            grid_dim[0] = self._grid_dim_x
        else:
            grid_dim[0] = int(self._grid_dim_x(**{
                tp_name: configuration[tp_name] for tp_name in self._grid_dim_x_tps
            }))
        if self._grid_dim_y_tps is None:
            grid_dim[1] = self._grid_dim_y
        else:
            grid_dim[1] = int(self._grid_dim_y(**{
                tp_name: configuration[tp_name] for tp_name in self._grid_dim_y_tps
            }))
        if self._grid_dim_z_tps is None:
            grid_dim[2] = self._grid_dim_z
        else:
            grid_dim[2] = int(self._grid_dim_z(**{
                tp_name: configuration[tp_name] for tp_name in self._grid_dim_z_tps
            }))
        block_dim = [1, 1, 1]
        if self._block_dim_x_tps is None:
            block_dim[0] = self._block_dim_x
        else:
            block_dim[0] = int(self._block_dim_x(**{
                tp_name: configuration[tp_name] for tp_name in self._block_dim_x_tps
            }))
        if self._block_dim_y_tps is None:
            block_dim[1] = self._block_dim_y
        else:
            block_dim[1] = int(self._block_dim_y(**{
                tp_name: configuration[tp_name] for tp_name in self._block_dim_y_tps
            }))
        if self._block_dim_z_tps is None:
            block_dim[2] = self._block_dim_z
        else:
            block_dim[2] = int(self._block_dim_z(**{
                tp_name: configuration[tp_name] for tp_name in self._block_dim_z_tps
            }))

        # prepare kernel arguments
        args = []
        for idx, inp in self._inputs.items():
            if isinstance(inp, numpy.ndarray):
                args.append(numpy.array([int(self._cu_device_ptr[idx])], dtype=numpy.uint64))
            else:
                args.append(numpy.array([inp], dtype=inp.dtype))
        args = numpy.array([arg.ctypes.data for arg in args], dtype=numpy.uint64)

        # warmups
        for _ in range(self._warmups):
            self._cpy_to_device()
            self._safe_call(cuda.cuLaunchKernel(kernel, *grid_dim, *block_dim, 0, self._cu_stream, args.ctypes.data, 0),
                            'failed to launch kernel', meta_data)

        # evaluations
        avg_runtime = 0.0
        for e in range(self._evaluations):
            self._cpy_to_device()
            pre_kernel_event = self._safe_call(cuda.cuEventCreate(0), 'failed to create pre-kernel event', meta_data)
            self._objects_to_free_on_error[pre_kernel_event] = cuda.cuEventDestroy
            post_kernel_event = self._safe_call(cuda.cuEventCreate(0), 'failed to create post-kernel event', meta_data)
            self._objects_to_free_on_error[post_kernel_event] = cuda.cuEventDestroy
            self._safe_call(cuda.cuEventRecord(pre_kernel_event, self._cu_stream),
                            'failed to record pre-kernel event', meta_data)
            self._safe_call(cuda.cuLaunchKernel(kernel, *grid_dim, *block_dim, 0, self._cu_stream, args.ctypes.data, 0),
                            'failed to launch kernel', meta_data)
            self._safe_call(cuda.cuEventRecord(post_kernel_event, self._cu_stream),
                            'failed to record post-kernel event', meta_data)
            self._safe_call(cuda.cuEventSynchronize(post_kernel_event),
                            'failed to synchronize with post-kernel event', meta_data)
            avg_runtime += self._safe_call(cuda.cuEventElapsedTime(pre_kernel_event, post_kernel_event),
                                           'failed to get elapsed time between kernel events', meta_data) * 1000000
            del self._objects_to_free_on_error[post_kernel_event]
            self._safe_call(cuda.cuEventDestroy(post_kernel_event),
                            'failed to destroy post-kernel event', meta_data)
            del self._objects_to_free_on_error[pre_kernel_event]
            self._safe_call(cuda.cuEventDestroy(pre_kernel_event),
                            'failed to destroy pre-kernel event', meta_data)
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
                        self._free_objects_before_error()
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
                            self._free_objects_before_error()
                            raise CostFunctionError(meta_data)
                meta_data['result_check'] = {'status': 'success', 'checked_inputs': tuple(self._gold_data.keys())}
        avg_runtime /= self._evaluations
        avg_runtime = float(ceil(avg_runtime))  # runtime can be ceiled, since nanoseconds are precise enough

        # free program resources
        del self._objects_to_free_on_error[module]
        self._safe_call(cuda.cuModuleUnload(module), 'failed to unload module', meta_data)
        del self._objects_to_free_on_error[nvrtc_program]
        self._safe_call(nvrtc.nvrtcDestroyProgram(nvrtc_program), 'failed to destroy NVRTC program', meta_data)

        return avg_runtime, meta_data

    def _init_device(self):
        if self._device_id is not None:
            self._cu_device = self._safe_call(cuda.cuDeviceGet(self._device_id), 'failed to retrieve device handle')
            if not self._silent:
                device_name = self._safe_call(cuda.cuDeviceGetName(1024, self._cu_device), 'failed to get device name')
                device_name = str(device_name, encoding='ascii').strip()[:-1]
                print(f'selecting CUDA device {self._device_id}: {device_name}')
            self._cu_context = self._safe_call(cuda.cuCtxCreate(0, self._cu_device), 'failed to create context')
            self._cu_stream = self._safe_call(cuda.cuStreamCreate(0), 'failed to create stream')

    def _alloc_buffers(self):
        if self._cu_context is not None:
            self._cu_device_ptr = {
                idx: self._safe_call(cuda.cuMemAlloc(inp.nbytes), f'failed to allocate CUDA device memory for input {idx}')
                for idx, inp in self._inputs.items() if isinstance(inp, numpy.ndarray)
            }

    def _cpy_to_device(self):
        if self._cu_stream is not None:
            for idx, deviceptr in self._cu_device_ptr.items():
                ndarray = self._inputs[idx]
                self._safe_call(cuda.cuMemcpyHtoDAsync(deviceptr, ndarray.ctypes.data, ndarray.nbytes, self._cu_stream),
                                f'failed to copy data from host to CUDA device memory for input {idx}')
            self._safe_call(cuda.cuStreamSynchronize(self._cu_stream),
                            f'failed to synchronize after host to device copy')

    def _cpy_to_host(self, host_memory: Dict[int, numpy.ndarray] = None):
        if self._cu_stream is not None:
            if host_memory is None:
                host_memory = self._inputs
            for idx, host_data in host_memory.items():
                if isinstance(host_data, numpy.ndarray):
                    self._safe_call(cuda.cuMemcpyDtoHAsync(host_data.ctypes.data, self._cu_device_ptr[idx],
                                                           host_data.nbytes, self._cu_stream),
                                    f'failed to copy data from CUDA device memory to host for input {idx}')
            self._safe_call(cuda.cuStreamSynchronize(self._cu_stream),
                            f'failed to synchronize after device to host copy')

    def _free_buffers(self):
        for idx, deviceptr in self._cu_device_ptr.items():
            self._safe_call(cuda.cuMemFree(deviceptr), f'failed to free CUDA device memory for input {idx}')
        self._cu_device_ptr.clear()

    def _free_device(self):
        if self._cu_stream is not None:
            self._safe_call(cuda.cuStreamDestroy(self._cu_stream), 'failed to destroy stream')
            self._cu_stream = None
        if self._cu_context is not None:
            self._safe_call(cuda.cuCtxDestroy(self._cu_context), 'failed to destroy context')
            self._cu_context = None
        self._cu_device = None
