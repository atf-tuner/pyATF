Cost Functions
==============

For high flexibility, the pyATF user can use any arbitrary, self-implemented cost function. pyATF allows as a cost function any Python function that takes as input a configuration of tuning parameters and returns a value of type :code:`pyatf.tuning_data.Cost` (currently defined as :code:`float`). pyATF interprets the cost function’s return value as the cost that has to be minimized during the auto-tuning process.

A cost function can raise the error :code:`pyatf.tuning_data.CostFunctionError` if the configuration to measure is invalid (e.g. because the configuration crashes due to too excessive memory usage) — the configuration is then penalized by the search technique (e.g., with a penalty cost). If any other error is raised, the tuning run is aborted by pyATF.

Pre-Implemented Cost Functions
------------------------------

pyATF provides the following pre-implemented cost functions:

.. py:class:: pyatf.cost_functions.generic.CostFunction

  .. py:function:: CostFunction(run_command: str)

    :param run_command: Run command (executed via :code:`subprocess.run`).

  .. py:function:: compile_command(compile_command: str)

    :param compile_command: Compile command (executed via :code:`subprocess.run`).

  .. py:function:: cost_file(costfile: str)

    :param cost_file: Path to cost file containing cost as string (must be convertible to :code:`pyatf.tuning_data.Cost`).

.. py:class:: pyatf.cost_functions.opencl.CostFunction

  .. py:function:: CostFunction(kernel: pyatf.cost_functions.opencl.Kernel)

    Initializes cost function with OpenCL kernel to tune.

  .. py:function:: platform_id(platform_id: int)

    Target OpenCL platform id.

  .. py:function:: device_id(device_id: int)

    Target OpenCL device id.

  .. py:function:: inputs(*inputs: Union[numpy.ndarray, numpy.generic])

    Kernel's input arguments (specified as instances of :code:`numpy.ndarray` and :code:`numpy.generic`).

  .. py:function:: global_size(gs_0: Union[int, Callable[..., int]], gs_1: Union[int, Callable[..., int]] = 1, gs_2: Union[int, Callable[..., int]] = 1)

    Kernel's 3-dimensional OpenCL global size as arithmetic expressions that may contain tuning parameters.

  .. py:function:: local_size(ls_0: Union[int, Callable[..., int]], ls_1: Union[int, Callable[..., int]] = 1, ls_2: Union[int, Callable[..., int]] = 1)

    Kernel's 3-dimensional OpenCL local size as arithmetic expressions that may contain tuning parameters.

  .. py:function:: check_result(index: int, gold_data_or_callable: Union[numpy.ndarray, numpy.generic, Callable], comparator = equality)

    Check result for scalar/buffer at position :code:`index` against :code:`gold_data_or_callable`.

    :param gold_data_or_callable: either of type: i) :code:`numpy.ndarray`, ii) :code:`numpy.generic`, or iii) a callable using kernel's input scalars/buffers (of type :code:`numpy.generic`/:code:`numpy.ndarray`) to compute a gold scalar/buffer.

    :param comparator: used for comparing kernel values against gold values; is a callable that takes two values as input (kernel and gold value) and returns True, iff the values are considered the same.

  .. py:function:: warmups(warmups: int)

    Number of warmups for each kernel run.

  .. py:function:: evaluations(evaluations: int)

    Number of evaluations for each kernel run.

  .. py:function:: silent(silent: bool)

    Silences log messages.

.. py:class:: pyatf.cost_functions.cuda.CostFunction

  .. py:function:: CostFunction(kernel: pyatf.cost_functions.cuda.Kernel)

    Initializes cost function with CUDA kernel to tune.

  .. py:function:: device_id(device_id: int)

    Target CUDA device id.

  .. py:function:: inputs(*inputs: Union[numpy.ndarray, numpy.generic])

    Kernel's input arguments (specified as instances of :code:`numpy.ndarray` and :code:`numpy.generic`).

  .. py:function:: grid_dim(x: Union[int, Callable[..., int]], y: Union[int, Callable[..., int]] = 1, z: Union[int, Callable[..., int]] = 1)

    Kernel's 3-dimensional CUDA grid dimension as arithmetic expressions that may contain tuning parameters.

  .. py:function:: block_dim(x: Union[int, Callable[..., int]], y: Union[int, Callable[..., int]] = 1, z: Union[int, Callable[..., int]] = 1)

    Kernel's 3-dimensional CUDA block dimension as arithmetic expressions that may contain tuning parameters.

  .. py:function:: check_result(index: int, gold_data_or_callable: Union[numpy.ndarray, numpy.generic, Callable], comparator = equality)

    Check result for scalar/buffer at position :code:`index` against :code:`gold_data_or_callable`.

    :param gold_data_or_callable: either of type: i) :code:`numpy.ndarray`, ii) :code:`numpy.generic`, or iii) a callable using kernel's input scalars/buffers (of type :code:`numpy.generic`/:code:`numpy.ndarray`) to compute a gold scalar/buffer.

    :param comparator: used for comparing kernel values against gold values; is a callable that takes two values as input (kernel and gold value) and returns True, iff the values are considered the same.

  .. py:function:: warmups(warmups: int)

    Number of warmups for each kernel run.

  .. py:function:: evaluations(evaluations: int)

    Number of evaluations for each kernel run.

  .. py:function:: silent(silent: bool)

    Silences log messages.

Misc
----

.. py:class:: pyatf.cost_functions.opencl.Kernel

  .. py:function:: Kernel( source: str, name: str = "func", flags: Iterable[str] = None )

    OpenCL kernel wrapper.

    :param source: OpenCL source code as string; function :code:`pyatf.cost_functions.opencl.path( path: str )` can be used to extract source code from file

    :param name: kernel name

    :param flags: kernel flags

.. py:class:: pyatf.cost_functions.cuda.Kernel

  .. py:function:: Kernel( source: str, name: str = "func", flags: Iterable[str] = None )

    CUDA kernel wrapper.

    :param source: CUDA source code as string; function :code:`pyatf.cost_functions.cuda.path( path: str )` can be used to extract source code from file

    :param name: kernel name

    :param flags: kernel flags