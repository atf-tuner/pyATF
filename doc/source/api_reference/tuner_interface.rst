Tuner
=====

.. cpp:class:: tuner

  Represents a tuner, that can be customized via its member setter functions.

  .. cpp:function:: tuner()

    Default constructor.

  .. cpp:function:: tuning_parameters(tps...)

    Sets program's tuning parameters.

  .. cpp:function:: tuning_parameters(tp_groups...)

    Sets program's tuning parameters as independent parameter groups.

  .. cpp:function:: search_technique(const search_technique& search_technique)

    Sets the search technique for exploration.

  .. cpp:function:: silent(bool silent)

    Silences log messages.

  .. cpp:function:: log_file(const std::string &log_file)

    Sets path to logfile.

  .. cpp:function:: tune(cost_function &cost_function, const abort_condition &abort_condition)

    Tunes :code:`cost_function` until :code:`abort_condition` is met.

    Default abort condition: explore full search space.

  .. cpp:function:: make_step(cost_function &cost_function)

    Make one tuning step using :code:`cost_function`.

  .. cpp:function:: configuration get_configuration()

    Request next configuration to evaluate.

    Evaluated cost must be reported to tuner via function :code:`report_cost`.

  .. cpp:function:: report_cost(cost_t cost)

    Report evaluated cost of requested configuration.

    Configuration had to be requested via :code:`get_configuration`.

  .. cpp:function:: tuning_status get_tuning_status()

    Returns the tuning status object.

.. cpp:class:: tuning_parameter

  .. cpp:function:: tuning_parameter(name, range, constraint)

    Specifies a tuning parameter by its name, range, and constraint.

    :param name: Tuning parameter's name as string
    :param range: Either: 1) :code:`atf::interval<T>(min, max)` which is an interval of values of type :code:`T` between :code:`min` and :code:`max` (both including); intervals may have as optional argument a :code:`step_size` and function :code:`generator` (for using values :code:`generator(min), ..., generator(max)`; currently pre-implemented: :code:`atf::pow_2`); 2) :code:`{ v_1, v_2, ... }` which is a set of values :code:`v_1`, :code:`v_2`, ... of same type :code:`T`

.. cpp:class:: configuration

  Configuration of tuning parameters (name-value pairs).

.. cpp:class:: tuning_status

  Tuning status object.

  .. cpp:function:: configuration best_configuration()

  .. cpp:function:: cost_t min_cost()

  .. cpp:function:: size_t number_of_evaluated_configs()

  .. cpp:function:: size_t number_of_invalid_configs()

  .. cpp:function:: size_t number_of_valid_configs()

  .. cpp:function:: size_t evaluations_required_to_find_best_found_result()

  .. cpp:function:: size_t valid_evaluations_required_to_find_best_found_result()

  .. cpp:function:: std::chrono::steady_clock::time_point tuning_start_time()