import math
from datetime import timedelta


def display_secs_as_time(secs):
    """
    Given a number of seconds, get a string in a human-readable time format, similar to
    hh:mm:ss.

    :param number secs:

    :return str:
    """
    # Get a number of seconds in h:mm:ss format.
    time_string = str(timedelta(seconds=secs))
    # If the precision is sub-second (i.e. there's a decimal), strip trailing 0's.
    if "." in time_string:
        time_string = time_string.rstrip("0")

    return time_string


def polar_coords(x, y):
    """
    From Euclidean x and y, calculate polar coordinates
        : r (distance from origin), and
        : theta (angle from x-axis, between -pi and pi).

    :return (r, theta):
    """
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta
