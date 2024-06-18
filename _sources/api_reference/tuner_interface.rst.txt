Tuner
=====

.. py:class:: pyatf.Tuner

  Represents a tuner, that can be customized via its member setter functions.

  .. py:function:: Tuner()

    Default constructor.

  .. py:function:: tuning_parameters(*tps: TP)

    Sets program's tuning parameters.

  .. py:function:: search_technique(search_technique: Union[SearchTechnique, SearchTechnique1D])

    Sets the search technique for exploration.

  .. py:function:: silent(silent: bool)

    Silences log messages.

  .. py:function:: log_file(log_file: str)

    Sets path to logfile.

  .. py:function:: tune(cost_function: CostFunction, abort_condition: Optional[AbortCondition] = None)

    Tunes :code:`cost_function` until :code:`abort_condition` is met.

    Default abort condition: explore full search space.

  .. py:function:: make_step(cost_function: CostFunction)

    Make one tuning step using :code:`cost_function`.

  .. py:function:: get_tuning_data() -> TuningData

    Returns the tuning data object.

.. py:class:: pyatf.TP

  .. py:function:: TP(name, range, constraint)

    Specifies a tuning parameter by its name, range, and constraint.

    :param name: Tuning parameter's name as string
    :param range: Either: 1) :code:`pyatf.Interval(min, max)` which is an interval of values between :code:`min` and :code:`max` (both including); intervals may have as optional argument a :code:`step_size` and function :code:`generator` (for using values :code:`generator(min), ..., generator(max)`; or 2) :code:`pyatf.Set(*values)`.

.. py:class:: pyatf.tuning_data.Configuration

  Configuration of tuning parameters (name-value pairs).

.. py:class:: pyatf.tuning_data.TuningData

  Tuning data object.

  .. py:attribute:: tuning_parameters (read-only)

  .. py:attribute:: constrained_search_space_size (read-only)

  .. py:attribute:: unconstrained_search_space_size (read-only)

  .. py:attribute:: search_space_generation_ns (read-only)

  .. py:attribute:: search_technique (read-only)

  .. py:attribute:: abort_condition (read-only)

  .. py:attribute:: tuning_start_timestamp (read-only)

  .. py:attribute:: terminated_early (read-only)

  .. py:attribute:: history (read-only)

  .. py:attribute:: improvement_history (read-only)

  .. py:attribute:: number_of_evaluated_configurations (read-only)

  .. py:attribute:: number_of_evaluated_valid_configurations (read-only)

  .. py:attribute:: number_of_evaluated_invalid_configurations (read-only)

  .. py:function:: total_tuning_duration()

  .. py:function:: configuration_of_min_cost()

  .. py:function:: search_space_coordinates_of_min_cost()

  .. py:function:: search_space_index_of_min_cost()

  .. py:function:: timestamp_of_min_cost()

  .. py:function:: duration_to_min_cost()

  .. py:function:: evaluations_to_min_cost()

  .. py:function:: valid_evaluations_to_min_cost()
