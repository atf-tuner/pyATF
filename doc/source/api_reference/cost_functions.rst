Cost Functions
==============

ATF allows as cost function any arbitrary C++ callable that takes as input a configuration of tuning parameters and returns a value for which operator :code:`<` is defined, e.g., :code:`size_t`.

Pre-Implemented Cost Functions
------------------------------

ATF provides the following pre-implemented cost functions:

.. cpp:class:: generic::cost_function

  .. cpp:function:: cost_function(const std::string &run_command)

    :param run_command: Run command (executed in bash).

  .. cpp:function:: compile_command(const std::string &compile_command)

    :param compile_command: Compile command (executed in bash).

  .. cpp:function:: costfile(const std::string &costfile)

    :param costfile: Path to costfile containing cost as string (must be convertible to :code:`cost_t`).

.. cpp:class:: opencl::cost_function

  .. cpp:function:: cost_function(const opencl::kernel &kernel)

    Initializes cost function with OpenCL kernel to tune.

  .. cpp:function:: platform_id(size_t platform_id)

    Target OpenCL platform id.

  .. cpp:function:: device_id(size_t device_id)

    Target OpenCL device id.

  .. cpp:function:: template<typename... Ts> inputs(Ts&&... inputs)

    Kernel's input arguments (specified as instances of :code:`atf::scalar<T>` and :code:`atf::buffer<T>`).

  .. cpp:function:: global_size(tp_int_expression&& gs_0, tp_int_expression&& gs_1 = 1, tp_int_expression&& gs_2 = 1)

    Kernel's 3-dimensional OpenCL global size as arithmetic expressions that may contain tuning parameters.

  .. cpp:function:: local_size(tp_int_expression&& ls_0, tp_int_expression&& ls_1 = 1, tp_int_expression&& ls_2 = 1)

    Kernel's 3-dimensional OpenCL local size as arithmetic expressions that may contain tuning parameters.

  .. cpp:function:: template<size_t index> check_result(gold_data, comparator = equality())

    Check result for scalar/buffer at position :code:`index` against :code:`gold_data`.

    :param gold_data: either of type: i) :code:`std::vector<T>` for :code:`atf::buffer<T>`, or ii) :code:`T` for :code:`atf::scalar<T>`

    :param comparator: used for comparing :code:`T` values; is of type :code:`std::function<bool(T,T)>`

  .. cpp:function:: template<size_t index> check_result(gold_callable, comparator = equality())

    Check result for scalar/buffer at position :code:`index` against scalar/buffer computed via :code:`gold_callable`.

    :param gold_callable: computes scalar/buffer (of type :code:`T`/:code:`std::vector<T>`) using kernel's input scalars/buffers (of type :code:`T`/:code:`std::vector<T>`)

    :param comparator: used for comparing :code:`T` values; is of type :code:`std::function<bool(T,T)>`

  .. cpp:function:: warmups(size_t warmups)

    Number of warmups for each kernel run.

  .. cpp:function:: evaluations(size_t evaluations)

    Number of evaluations for each kernel run.

.. cpp:class:: cuda::cost_function

  .. cpp:function:: cost_function(const cuda::kernel &kernel)

    Initializes cost function with CUDA kernel to tune.

  .. cpp:function:: device_id(int device_id)

    Target CUDA device id.

  .. cpp:function:: template<typename... Ts> inputs(Ts&&... inputs)

    Kernel's input arguments (specified as instances of :code:`atf::scalar<T>` and :code:`atf::buffer<T>`).

  .. cpp:function:: grid_dim(tp_int_expression&& gs_0, tp_int_expression&& gs_1 = 1, tp_int_expression&& gs_2 = 1)

    Kernel's 3-dimensional CUDA grid dimension as arithmetic expressions that may contain tuning parameters.

  .. cpp:function:: block_dim(tp_int_expression&& ls_0, tp_int_expression&& ls_1 = 1, tp_int_expression&& ls_2 = 1)

    Kernel's 3-dimensional CUDA block dimension as arithmetic expressions that may contain tuning parameters.

  .. cpp:function:: template<size_t index> check_result(gold_data, comparator = equality())

    Check result for scalar/buffer at position :code:`index` against :code:`gold_data`.

    :param gold_data: either of type: i) :code:`std::vector<T>` for :code:`atf::buffer<T>`, or ii) :code:`T` for :code:`atf::scalar<T>`

    :param comparator: used for comparing :code:`T` values; is of type :code:`std::function<bool(T,T)>`

  .. cpp:function:: template<size_t index> check_result(gold_callable, comparator = equality())

    Check result for scalar/buffer at position :code:`index` against scalar/buffer computed via :code:`gold_callable`.

    :param gold_callable: computes scalar/buffer (of type :code:`T`/:code:`std::vector<T>`) using kernel's input scalars/buffers (of type :code:`T`/:code:`std::vector<T>`)

    :param comparator: used for comparing :code:`T` values; is of type :code:`std::function<bool(T,T)>`

  .. cpp:function:: warmups(size_t warmups)

    Number of warmups for each kernel run.

  .. cpp:function:: evaluations(size_t evaluations)

    Number of evaluations for each kernel run.

Misc
----

.. cpp:class:: template<typename T> scalar

  .. cpp:function:: scalar(T value)

    Scalar representing :code:`value`.

  .. cpp:function:: scalar()

    Random scalar.

  .. cpp:function:: scalar(std::array<T, 2> interval)

    Random scalar in :code:`interval`.

    :param interval: interval's min and max values.

.. cpp:class:: template<typename T> buffer

  .. cpp:function:: buffer(std::vector<T> values)

    Buffer representing :code:`values`.

  .. cpp:function:: buffer(size_t size, T value)

    Buffer containing :code:`size`-many times :code:`value`.

  .. cpp:function:: buffer(size_t size)

    Random buffer of size :code:`size` containing values of type :code:`T`.

  .. cpp:function:: buffer(size_t size, std::array<T, 2> interval)

    Random buffer of size :code:`size` containing values of type :code:`T` in :code:`interval`.

    :param interval: interval's min and max values.

  .. cpp:function:: buffer(size_t size, std::function<T(size_t)> generator)

    Buffer containing values :code:`generator(0), generator(1), ... , generator(size - 1)`.

.. cpp:class:: template<typename... Ts> opencl::kernel

  :param Ts: Types of kernel's input arguments (specified as :code:`atf::scalar<T>` and :code:`atf::buffer<T>`).

  .. cpp:function:: kernel( std::string source, std::string name = "func", std::string flags = "" )

    OpenCL kernel wrapper.

    :param source: OpenCL source code as string; function :code:`atf::path( std::string path )` can be used to extract source code from file

    :param name: kernel name

    :param flags: kernel flags

.. cpp:class:: template<typename... Ts> cuda::kernel

  :param Ts: Types of kernel's input arguments (specified as :code:`atf::scalar<T>` and :code:`atf::buffer<T>`).

  .. cpp:function:: kernel( std::string source, std::string name = "func", std::string flags = "" )

    CUDA kernel wrapper.

    :param source: CUDA source code as string; function :code:`atf::path( std::string path )` can be used to extract source code from file

    :param name: kernel name

    :param flags: kernel flags