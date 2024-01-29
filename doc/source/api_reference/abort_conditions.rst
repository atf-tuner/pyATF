Abort Conditions
================

.. py:class:: pyatf.abort_conditions.abort_condition.AbortCondition

  .. py:function:: stop(tuning_data: TuningData) -> bool

    Determines whether a tuning run should be stopped based on its tuning data.

    :param  tuning_data: The current data of the tuning run (best found configuration so far, tuning time, ...)
    :return: true, if the tuning should stop, false otherwise