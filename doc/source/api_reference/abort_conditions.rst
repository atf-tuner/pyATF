Abort Conditions
================

.. cpp:class:: abort_condition

  .. cpp:function:: bool stop(const tuning_status& status)

    Determines whether a tuning run should be stopped based on its tuning status.

    :param  status: The current status of the tuning run (best found configuration so far, tuning time, ...)
    :return: true, if the tuning should stop, false otherwise