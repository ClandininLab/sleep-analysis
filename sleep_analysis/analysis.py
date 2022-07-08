""" Module for analysis methods
"""

from datetime import datetime, timedelta
from warnings import warn

import numpy as np
import pandas as pd
import pingouin as pg

from sleep_analysis.constants import SLEEP_MIN_LEN
from sleep_analysis.utils import (assert_df_equals, contiguous_regions,
                                  get_lights_on_datetimes,
                                  get_long_light_cycles,
                                  get_short_light_cycles,
                                  get_well_formed_light_cycles)

pd.options.display.width = 120


def summarize(data_df):
    """Summarize data, report missing data

    Args:
      data_df: as returned by sleep_analysis.data.read_data - pd.DataFrame
    """
    _data_df = data_df.copy(deep=True)

    # ==========================================================================
    #                            GENERAL SUMMARY
    # ==========================================================================

    # TODO: add activity descriptions, number of complete zg days

    print("General summary:\n")
    print(f"\t Data ranges from {data_df.index[0]} to {data_df.index[-1]}\n")
    print(f"\t {len(data_df)} timepoints\n")
    print(f"\t {len(data_df.columns) -1} flies\n")
    print("\n")

    # ==========================================================================
    #                           MISSING TIMEPOINTS
    # ==========================================================================

    t_deltas = np.unique(np.diff(data_df.index))

    if len(t_deltas) == 1:
        print("No missing timepoints\n")
    else:
        start_time = data_df.index[0]
        end_time = data_df.index[-1]
        t_delta = t_deltas[0]

        tps = [
            start_time + t_delta * idx
            for idx in range(int((end_time - start_time) / t_delta) + 1)
        ]
        missing_tps = sorted(set(tps) - set(data_df.index))

        if len(missing_tps) > 10:
            print(
                f"Missing {len(missing_tps)} timepoints between {missing_tps[0]} and {missing_tps[-1]}\n"
            )
        else:
            print("Missing timepoints:\n")
            for missing_tp in missing_tps:
                print(f"\t {missing_tp}\n")

    print("\n")

    # ==========================================================================
    #                            LIGHT CYCLE
    # ==========================================================================

    print("Light cycle:\n")

    too_short = get_short_light_cycles(data_df)
    too_long = get_long_light_cycles(data_df)
    well_formed = np.array(get_well_formed_light_cycles(data_df))

    print(
        f"\t {len(well_formed)} well-formed cycles of average length {np.mean(well_formed[:, 1] - well_formed[:, 0])}\n"
    )

    print(f"\t {len(too_short) + len(too_long)} malformed cycles:\n")

    print(f"\t\t {len(too_short)} too short:\n")

    for start_time, end_time in too_short:
        print(
            f"\t\t\t Cycle of length {end_time - start_time} starting at {start_time}\n"
        )

    print(f"\t\t {len(too_long)} too long:\n")

    for start_time, end_time in too_long:
        print(
            f"\t\t\t Cycle of length {end_time - start_time} starting at {start_time}\n"
        )

    assert_df_equals(data_df, _data_df)


def mortality_filter(
    data_df, surviving_flies=None, expected_quiescence_period=timedelta(hours=12)
):
    """Removes columns for flies that did not survive the trial

     Can either explicitly specify columns of surviving flies (useful for manual annotation)
       or estimate the expected maximum quescience period, in which case fly mortality
       is inferred from activity

     Args:
       data_df: as returned by sleep_analysis.data.read_data - pd.DataFrame
       surviving_flies: (optional) column names of flies that survived the whole trial - [str]
       expected_quescience_period: estimated period of dormancy beyond which flies will be considered dead - timedelta
         does not align to lights on

    Returns:
      filtered_df - pd.DataFrame
    """
    assert isinstance(data_df, pd.DataFrame)
    assert isinstance(surviving_flies, list) or isinstance(
        expected_quiescence_period, timedelta
    )

    fly_cols = [col for col in data_df.columns if col.startswith("fly")]

    if surviving_flies is None:
        print("Using mortality heuristic")

        surviving_flies = []

        for col in fly_cols:
            est_tod = estimate_time_of_death(data_df, col)

            if data_df.index[-1] - est_tod < expected_quiescence_period:
                surviving_flies.append(col)

    else:
        print("Using mortality annotations")

    print(f"Pruning {set(fly_cols) - set(surviving_flies)}")

    return data_df[["light"] + surviving_flies]


def make_zeitgeber_df(data_df, include_incomplete=False, timedelta_workaround=False):
    """Constructs a zeitgeber DataFrame

     Data is trimmed from the first identifiable zeitgeber (lights on) to the last

    Args:
      data_df: as returned by sleep_analysis.data.read_data - pd.DataFrame
      include_incomplete: (optional) if True, zero pads to include incomplete data at start - bool
      timedelta_workaround: (optional) workaround to fix overflow bug when plotting timedeltas

    Returns:
      zg_df - pd.DataFrame

    """
    assert isinstance(data_df, pd.DataFrame)

    # construct multindex
    multi_index = []
    lights_on = get_lights_on_datetimes(data_df)

    if data_df["light"][0]:
        lights_on = lights_on[1:]

        if not include_incomplete:
            print(
                f"Discarding first {lights_on[0] - data_df.index[0]}, last {data_df.index[-1] - lights_on[-1]}"
            )

    too_short = get_short_light_cycles(data_df)
    too_long = get_long_light_cycles(data_df)

    # validate that light cycle is nearly 24 hours
    if too_short or too_long:
        warn(
            (
                "Achtung! Malformed light column! make_zeitgeber_df expects nearly 24 hour light cycles."
                " See summary for more info"
            )
        )

    if include_incomplete:
        # prefix
        prefix_start_time = lights_on[0] - timedelta(hours=24)

        prefix_new_times = []
        for idx in range(60 * 24):
            time = prefix_start_time + timedelta(minutes=idx)

            if time not in data_df.index:
                prefix_new_times.append(time)
            else:
                break

        prefix_pad_df = pd.DataFrame(
            np.nan, index=prefix_new_times, columns=data_df.columns
        )

        # suffix
        suffix_start_time = lights_on[-1]

        suffix_new_times = []
        for idx in range(60 * 24):
            time = suffix_start_time + timedelta(minutes=idx)

            if time not in data_df.index:
                suffix_new_times.append(time)

        suffix_pad_df = pd.DataFrame(
            np.nan, index=suffix_new_times, columns=data_df.columns
        )

        data_df = pd.concat([prefix_pad_df, data_df, suffix_pad_df])

        lights_on = (
            [prefix_start_time] + lights_on + [suffix_start_time + timedelta(hours=24)]
        )

    for day_idx, (zg_start, zg_end) in enumerate(zip(lights_on[:-1], lights_on[1:])):
        timestamps = data_df[zg_start:zg_end].index[:-1]

        if timedelta_workaround:
            timedeltas = [_td_workaround(ts - timestamps[0]) for ts in timestamps]
        else:
            timedeltas = [ts - timestamps[0] for ts in timestamps]

        multi_index.extend([(day_idx, zg_td) for zg_td in timedeltas])

    # trim to match length of multi index
    zg_df = data_df[lights_on[0] : lights_on[-1]][:-1]

    # set index
    zg_df.index = pd.MultiIndex.from_tuples(multi_index, names=["day", "zg_time"])

    return zg_df


def total_sleep(zg_df, start_time=None, end_time=None, start_day=None, end_day=None):
    """Calculate daily total sleep within time period

    Args:
      zg_df: zeitgeber df - pd.DataFrame
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      start_day: (optional) by default all days are used - int
      end_day: (optional) by default all days are used - int

    Returns:
      total_sleep_df - pd.DataFrame
    """
    _zg_df = zg_df.copy(deep=True)

    # validate times
    start_time = start_time or timedelta()
    end_time = end_time or timedelta(hours=24)

    assert isinstance(start_time, timedelta)
    assert isinstance(end_time, timedelta)
    assert start_time < end_time

    # validate days
    start_day = start_day or 0
    end_day = end_day or max(zg_df.index.get_level_values(0))

    assert isinstance(start_day, int)
    assert isinstance(end_day, int)
    assert start_day <= end_day

    fly_cols = [col for col in zg_df.columns if col.startswith("fly")]

    idx = pd.IndexSlice
    sleep_df = sleep_filter(zg_df).loc(axis=0)[
        idx[start_day:end_day, start_time:end_time]
    ]

    total_sleep_df = sleep_df[fly_cols].groupby(level=[0]).sum()

    assert_df_equals(zg_df, _zg_df)

    return total_sleep_df


def cumulative_bout_duration(
    zg_dfs, labels, start_time=None, end_time=None, start_day=None, end_day=None
):
    """Calculate total duration of bouts per fly averaged over all days

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
     labels: list of labels corresponding to dataframes, one for each genotype - [str]
     start_time: (optional) zeitgeber start time - timedelta
     end_time: (optional) zeitgeber end time - timedelta
     start_day: (optional) by default all days are used - int
     end_day: (optional) by default all days are used - int

    Returns:
      bout_len_df
    """
    return group_by_light(
        zg_dfs,
        labels,
        "bout_len",
        start_time=start_time,
        end_time=end_time,
        start_day=start_day,
        end_day=end_day,
    )


def cumulative_bout_number(
    zg_dfs, labels, start_time=None, end_time=None, start_day=None, end_day=None
):
    """Calculate total number of bouts per fly averaged over all days

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
     labels: list of labels corresponding to dataframes, one for each genotype - [str]
     start_time: (optional) zeitgeber start time - timedelta
     end_time: (optional) zeitgeber end time - timedelta
     start_day: (optional) by default all days are used - int
     end_day: (optional) by default all days are used - int

    Returns:
      bout_len_df
    """
    return group_by_light(
        zg_dfs,
        labels,
        "bout_num",
        start_time=start_time,
        end_time=end_time,
        start_day=start_day,
        end_day=end_day,
    )


def cumulative_sleep(
    zg_dfs, labels, start_time=None, end_time=None, start_day=None, end_day=None
):
    """Calculate total sleep per fly averaged over all days

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
     labels: list of labels corresponding to dataframes, one for each genotype - [str]
     start_time: (optional) zeitgeber start time - timedelta
     end_time: (optional) zeitgeber end time - timedelta
     start_day: (optional) by default all days are used - int
     end_day: (optional) by default all days are used - int

    Returns:
      cumulative_sleep_df
    """
    return group_by_light(
        zg_dfs,
        labels,
        "sleep",
        start_time=start_time,
        end_time=end_time,
        start_day=start_day,
        end_day=end_day,
    )


def cumulative_activity(
    zg_dfs, labels, start_time=None, end_time=None, start_day=None, end_day=None
):
    """Calculate total activity per fly averaged over all days

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
     labels: list of labels corresponding to dataframes, one for each genotype - [str]
     start_time: (optional) zeitgeber start time - timedelta
     end_time: (optional) zeitgeber end time - timedelta
     start_day: (optional) by default all days are used - int
     end_day: (optional) by default all days are used - int

    Returns:
      cumulative_activity_df
    """
    return group_by_light(
        zg_dfs,
        labels,
        "activity",
        start_time=start_time,
        end_time=end_time,
        start_day=start_day,
        end_day=end_day,
    )


def group_by_light(
    zg_dfs,
    labels,
    quantity,
    start_time=None,
    end_time=None,
    start_day=None,
    end_day=None,
):
    """Calculates quantity and groups by light, returning average (per day) of quantity by light by genotype

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
     labels: list of labels corresponding to dataframes, one for each genotype - [str]
     quantity: one of sleep, bout_num, bout_len, activity - str
     start_time: (optional) zeitgeber start time - timedelta
     end_time: (optional) zeitgeber end time - timedelta
     start_day: (optional) by default all days are used - int
     end_day: (optional) by default all days are used - int

    Returns:
      grouped_df - pd.DataFrame

    """
    # validate times
    start_time = start_time or timedelta()
    end_time = end_time or timedelta(hours=24)

    assert isinstance(start_time, timedelta)
    assert isinstance(end_time, timedelta)
    assert start_time < end_time

    # validate days
    start_day = start_day or 0
    end_day_set = end_day is not None

    assert isinstance(start_day, int)

    if end_day is not None:
        assert isinstance(end_day, int)
        assert start_day < end_day

    assert quantity in ["sleep", "bout_num", "bout_len", "activity"]

    melted_dfs = []

    for trials, label in zip(zg_dfs, labels):
        for trial_df in trials:
            _trial_df = trial_df.copy(deep=True)

            if not end_day_set:
                end_day = max(trial_df.index.get_level_values(0))

            if quantity == "sleep":
                _trial_df = sleep_filter(_trial_df)

                idx = pd.IndexSlice
                _trial_df = _trial_df.loc(axis=0)[
                    idx[start_day:end_day, start_time:end_time]
                ]

                # NOTE: assumes no missing timepoints and 1 minute sampling interval
                trial_length = len(_trial_df) / (
                    ((end_time - start_time).days * 1440)
                    + ((end_time - start_time).seconds / 60)
                )
                melted_df = (_trial_df.groupby("light").sum() / trial_length).T.melt()

            if quantity == "bout_num":
                _trial_df = bout_filter(trial_df)

                idx = pd.IndexSlice
                _trial_df = _trial_df.loc(axis=0)[
                    idx[start_day:end_day, start_time:end_time]
                ]

                fly_cols = [col for col in _trial_df.columns if col.startswith("fly")]
                _trial_df[fly_cols] = _trial_df[fly_cols] != 0

                # NOTE: assumes no missing timepoints and 1 minute sampling interval
                trial_length = len(_trial_df) / (
                    ((end_time - start_time).days * 1440)
                    + ((end_time - start_time).seconds / 60)
                )
                melted_df = (_trial_df.groupby("light").sum() / trial_length).T.melt()

            if quantity == "bout_len":
                _trial_df = bout_filter(trial_df)

                idx = pd.IndexSlice
                _trial_df = _trial_df.loc(axis=0)[
                    idx[start_day:end_day, start_time:end_time]
                ]

                cum_len = _trial_df.groupby("light").sum().T.melt()
                _trial_df.loc[:, _trial_df.columns != "light"] = (
                    _trial_df.loc[:, _trial_df.columns != "light"] != 0
                )
                cum_num = _trial_df.groupby("light").sum().T.melt()

                cum_len["value"] = cum_len["value"] / cum_num["value"]
                melted_df = cum_len

            if quantity == "activity":
                idx = pd.IndexSlice
                _trial_df = _trial_df.loc(axis=0)[
                    idx[start_day:end_day, start_time:end_time]
                ]

                # NOTE: assumes no missing timepoints and 1 minute sampling interval
                trial_length = len(_trial_df) / (
                    ((end_time - start_time).days * 1440)
                    + ((end_time - start_time).seconds / 60)
                )
                melted_df = (_trial_df.groupby("light").sum() / trial_length).T.melt()

            melted_df["label"] = label

            melted_dfs.append(melted_df)

    grouped_df = pd.concat(melted_dfs, ignore_index=True)
    # NOTE: astype(object) is a workaround for https://github.com/pandas-dev/pandas/issues/23305
    # unclear how the the workaround works
    grouped_df = grouped_df.astype(object).replace([0, 1], ["dark", "light"])

    return grouped_df.rename(columns={"value": quantity})


def sleep_filter(df, min_len=SLEEP_MIN_LEN):
    """Replaces fly column values with 1 where there was a consecutive run of min_len or more 0s, 0 elsewhere
    Does not modify column names

    Args:
      df: must contain fly columns, assumes there are no missing timepoints - pd.DataFrame
      min_len: (optional) defines sleep bouts - int
        min number of consecutive timepoints of inactivity that define a sleep bout

    Returns:
      sleep_df - pd.DataFrame

    """
    sleep_df = df.copy(deep=True)

    fly_cols = [col for col in sleep_df.columns if col.startswith("fly")]

    # df must have at least one fly col
    assert len(fly_cols)

    for fly_col in fly_cols:
        labels = (sleep_df[fly_col] != sleep_df[fly_col].shift()).cumsum()
        # mask of runs of more than min_len consecutive values
        run_mask = labels.map(labels.value_counts()) >= min_len
        # mask of zeros
        value_mask = sleep_df[fly_col] == 0

        sleep_df.loc[:, fly_col] = (run_mask & value_mask).astype(int)

    return sleep_df


# TODO: optimize with some logic used for sleep_filter
def bout_filter(df, min_len=5):
    """Replaces fly columns values. The value at the start of each bout is the
    length of that bout, zero elsewhere
    Does not modify column names

    Args:
      df: must contain fly columns, assumes there are no missing timepoints - pd.DataFrame
      min_len: (optional) defines bouts - int
        a run of adjacent timepoints with no activity >= min_len will be considered a bout

    Returns:
      bout_df - pd.DataFrame

    """
    assert isinstance(df, pd.DataFrame)

    # make a deep copy to avoid side-effects
    bout_df = df.copy(deep=True)

    fly_cols = [col for col in bout_df.columns if col.startswith("fly")]

    # df must have at least one fly_col
    assert len(fly_cols)

    for fly_col in fly_cols:
        bout_starts = []
        bout_lens = []

        for interval in contiguous_regions((bout_df.loc[:, fly_col] == 0).to_numpy()):
            if interval[1] - interval[0] >= min_len:
                bout_starts.append(interval[0])
                bout_lens.append(interval[1] - interval[0])

        bout_df[fly_col] = 0

        for bout_idx, bout_len in zip(bout_starts, bout_lens):
            bout_df.iloc[bout_idx, bout_df.columns.get_indexer([fly_col])] = bout_len

    return bout_df


def estimate_time_of_death(data_df, fly_col):
    """Estimates time of death for a fly
    Manual annotation is recommended for final analysis

    Args:
      data_df - pd.DataFrame
      fly_df - str

    Returns:
      est_time_of_death - datetime
    """
    _data_df = data_df.copy(deep=True)

    # simply finds start index of last contiguous region of no activity
    # sensitive to noise
    # a more sophisticated method is warranted
    death_idx = contiguous_regions(data_df[fly_col] == 0)[-1][0]

    est_time_of_death = datetime.fromtimestamp(
        data_df[fly_col].index[death_idx].timestamp()
    )

    assert_df_equals(data_df, _data_df)

    return est_time_of_death


def _td_workaround(t_delta):
    """Ugly workaround for overflow bug when formatting timedeltas for plotting"""
    return datetime(
        year=10,
        month=1,
        day=1,
        hour=t_delta.seconds // 3600,
        minute=(t_delta.seconds // 60) % 60,
    )


def anova_assistant(zg_dfs, labels, start_time, end_time):
    """Runs the flowchart for a slice for cumulative activity, sleep, bout duration and bout number
    Slice cannot straddle terminator

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
     labels: list of labels corresponding to dataframes, one for each genotype - [str]
     start_time: (optional) zeitgeber start time - timedelta
     end_time: (optional) zeitgeber end time - timedelta

    """
    print("=" * 80)

    print("Activity:\n")

    _anova_flowchart(
        group_by_light(zg_dfs, labels, "activity", start_time, end_time), "activity"
    )

    print("=" * 80)

    print("Sleep:\n")

    _anova_flowchart(
        group_by_light(zg_dfs, labels, "sleep", start_time, end_time), "sleep"
    )

    print("=" * 80)

    print("Bout duration:\n")

    _anova_flowchart(
        group_by_light(zg_dfs, labels, "bout_len", start_time, end_time), "bout_len"
    )

    print("=" * 80)

    print("Bout number:\n")

    _anova_flowchart(
        group_by_light(zg_dfs, labels, "bout_num", start_time, end_time), "bout_num"
    )

    print("=" * 80)


def _anova_flowchart(cumulative_df, quantity):
    assert len(cumulative_df["light"].unique()) == 1, "Slice cannot straddle terminator"

    # equal variance case
    if pg.homoscedasticity(data=cumulative_df, dv=quantity, group="label").loc[
        "levene", "equal_var"
    ]:
        print(" Equal variance case:\n\n")

        print("  Anova:\n")
        print(pg.anova(data=cumulative_df, dv=quantity, between="label"))

        print("\n  Tukey post-hoc:\n")
        print(pg.pairwise_tukey(data=cumulative_df, dv=quantity, between="label"))

    # unequal variance case
    else:
        print(" Unequal variance case:\n\n")

        print("  Welch Anova:\n")
        print(pg.welch_anova(data=cumulative_df, dv=quantity, between="label"))

        print("\n  Games-Howell post-hoc:\n")
        print(pg.pairwise_gameshowell(data=cumulative_df, dv=quantity, between="label"))
