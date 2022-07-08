""" Module for definition of global constants
"""

from datetime import datetime

# min number of consecutive timepoints of inactivity that define a sleep bout
SLEEP_MIN_LEN = 5

# timezone not handled, none of our data has timezones
UNIX_EPOCH = datetime(1970, 1, 1)
