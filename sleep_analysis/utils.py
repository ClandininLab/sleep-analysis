""" Module for misc. utility methods
"""

from datetime import datetime, timedelta
from warnings import warn

import numpy as np
import pandas as pd


def contiguous_regions(condition):
    """Finds contiguous True regions of the 1D boolean array "condition".
    Returns a 2D array where the first column is the start index of the region
    and the second column is the end index.

    Adapted from https://stackoverflow.com/questions/22592764/numpy-detection-of-region-borders
    """
    # Find the indicies of changes in "condition"
    idx = np.flatnonzero(np.diff(condition)) + 1

    # Prepend or append the start or end indicies to "idx"
    # if there's a block of "True"'s at the start or end...
    if condition[0]:
        idx = np.append(0, idx)
    if condition[-1]:
        idx = np.append(idx, len(condition))

    return idx.reshape(-1, 2)


def get_short_light_cycles(data_df):
    """ Return list of intervals for light cycles that are less than 24 hours

    Args:
      data_df - pd.DataFrame

    Returns:
      light_cycles: list of start and stop times - [(datetime, datetime)]
    """
    lights_on = get_lights_on_datetimes(data_df)

    if data_df['light'][0]:
        lights_on = lights_on[1:]

    too_short = np.diff(lights_on) < timedelta(hours=23, minutes=55)

    light_cycles = [
        (start_time, end_time) for start_time, end_time, cond in zip(lights_on[:-1], lights_on[1:], too_short) if cond
    ]

    return light_cycles


def get_long_light_cycles(data_df):
    """ Return list of intervals for light cycles that are less than 24 hours

    Args:
      data_df - pd.DataFrame

    Returns:
      light_cycles: list of start and stop times - [(datetime, datetime)]
    """
    lights_on = get_lights_on_datetimes(data_df)

    if data_df['light'][0]:
        lights_on = lights_on[1:]

    too_long = np.diff(lights_on) > timedelta(hours=24, minutes=5)

    light_cycles = [
        (start_time, end_time) for start_time, end_time, cond in zip(lights_on[:-1], lights_on[1:], too_long) if cond
    ]

    return light_cycles


def get_well_formed_light_cycles(data_df):
    """ Return list of intervals for light cycles that are approximately 24 hours

    Args:
      data_df - pd.DataFrame

    Returns:
      light_cycles: list of start and stop times - [(datetime, datetime)]
    """
    lights_on = get_lights_on_datetimes(data_df)

    if data_df['light'][0]:
        lights_on = lights_on[1:]

    well_formed = np.logical_and(
        timedelta(hours=23, minutes=55) < np.diff(lights_on),
        np.diff(lights_on) < timedelta(hours=24, minutes=5)
    )

    light_cycles = [
        (start_time, end_time) for start_time, end_time, cond in zip(lights_on[:-1], lights_on[1:], well_formed) if cond
    ]

    return light_cycles


def get_lights_on_datetimes(data_df):
    """ Extracts list of times that the lights turned on everyday

    Ignores the first interval if lights are on at the start of the data

    Args:
      data_df: as returned by sleep_analysis.data.read_data - pd.DataFrame

    Returns:
      lights_on - [datetime]
    """
    assert isinstance(data_df, pd.DataFrame)

    region_idxs = contiguous_regions(data_df['light'])

    return [data_df.index[start_idx] for start_idx, _ in region_idxs]


def get_single_timepoint_glitches(data_df):
    """ Return list of indices where the light column changed sign for a single timepoint

    Args:
      data_df - pd.DataFrame

    Returns:
      glitch_idxs - [int]
    """
    glitches = []

    for idx in range(1, len(data_df) - 1):
        window = list(data_df.iloc[idx - 1:idx + 2]['light'])
        if window == [0, 1, 0] or window == [1, 0, 1]:
            glitches.append(idx)

    return glitches


def set_light_cycle(data_df, lights_on_time, cycle_length=timedelta(hours=24)):
    """ Set the light column in data_df to an artificial light cycle

    Args:
      data_df - pd.DataFrame
      lights_on_time: light cycle start time, time from midnight - timedelta
      cycle_length: length of light cycle - timedelta

    Returns:
      data_df - pd.DataFrame
    """
    assert 'light' in data_df.columns

    well_formed = np.array(get_well_formed_light_cycles(data_df))

    warn((
        f"Using cycle of length {cycle_length}. "
        f"Existing light cycle has average length of {np.mean(well_formed[:, 1] - well_formed[:, 0])} "
        f"over {len(well_formed)} well-formed cycles."
    ))

    half_cycle = cycle_length / 2

    # set all zeros then set lights on
    data_df['light'] = 0

    start_date = data_df.index[0]
    end_date = data_df.index[-1]

    lights_on_ptr = datetime(start_date.year, start_date.month, start_date.day) + lights_on_time

    while lights_on_ptr <= end_date:
        data_df.loc[lights_on_ptr: lights_on_ptr + half_cycle, 'light'] = 1

        lights_on_ptr += cycle_length

    return data_df


def assert_df_equals(df1, df2):
    """ Assert that df1 and df2 are identical
    """
    assert df1.equals(df2)
    assert (df1.columns == df2.columns).all()


def assert_data_equals(data1, data2):
    """ Assert canonical format multi-trial multi genotype data are identical
    """
    for gt_trials, _gt_trials in zip(data1, data2):
        for zg_df, _zg_df in zip(gt_trials, _gt_trials):
            assert_df_equals(zg_df, _zg_df)
