def equality(x, y):
    return x == y


def absolute_difference(max_diff):
    return lambda x, y: abs(x - y) <= max_diff
