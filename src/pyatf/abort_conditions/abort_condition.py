from pyatf.tuning_data import TuningData


class AbortCondition:
    def stop(self, tuning_data: TuningData):
        """
        Determines whether a tuning run should be stopped based on its tuning data.

        :param tuning_data: The current data of the tuning run (best found configuration so far, tuning time, ...)
        :return: True, if the tuning should stop, False otherwise
        """
        raise NotImplementedError

    def progress(self, tuning_data: TuningData):
        """
        Returns the current progress towards the abort condition or None, if the progress cannot be determined.

        :param tuning_data: The current data of the tuning run (best found configuration so far, tuning time, ...)
        :return: A value in [0,1] representing the progress percentage or None, if the progress cannot be determined.
        """
        return None

    def to_json(self):
        """
        Returns the abort condition in json format. Used for logging purposes only

        :return: The abort condition in json format.
        """
        return {'kind': type(self).__name__}


class And(AbortCondition):
    def __init__(self, *conditions: AbortCondition):
        self._conditions = conditions

    def stop(self, tuning_data: TuningData):
        if not self._conditions:
            return True
        stop = True
        for c in self._conditions:
            if not c.stop(tuning_data):
                stop = False
                break
        return stop

    def progress(self, tuning_data: TuningData):
        if not self._conditions:
            return None
        min_progress = 1.0
        for c in self._conditions:
            progress = c.progress(tuning_data)
            if progress is None:
                return None
            else:
                min_progress = min(min_progress, progress)
        return min_progress

    def to_json(self):
        return {'kind': 'And', 'conditions': list(
            c.to_json()
            for c in self._conditions
        )}


class Or(AbortCondition):
    def __init__(self, *conditions: AbortCondition):
        self._conditions = conditions

    def stop(self, tuning_data: TuningData):
        if not self._conditions:
            return True
        stop = False
        for c in self._conditions:
            if c.stop(tuning_data):
                stop = True
                break
        return stop

    def progress(self, tuning_data: TuningData):
        if not self._conditions:
            return None
        max_progress = 0.0
        for c in self._conditions:
            progress = c.progress(tuning_data)
            if progress is None:
                return None
            else:
                max_progress = max(max_progress, progress)
        return max_progress

    def to_json(self):
        return {'kind': 'Or', 'conditions': list(
            c.to_json()
            for c in self._conditions
        )}
