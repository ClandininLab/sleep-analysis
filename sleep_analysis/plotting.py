""" Module for plotting methods
"""

from datetime import datetime, timedelta
from warnings import warn

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from sleep_analysis.analysis import (group_by_light, make_zeitgeber_df,
                                     sleep_filter, total_sleep)
from sleep_analysis.constants import UNIX_EPOCH
from sleep_analysis.utils import (assert_data_equals, assert_df_equals,
                                  contiguous_regions)


def actogram(data_df, fly_col=None, zg_faithful=True):
    """Activity summary for a single fly or averaged across all flies.
    Leaving zg_faithful set to it's default value of True is recommended as it
    faithfully shows how the light column will be parsed by make_zeitgeber_df.
    Set it to False for a literal representation of a malformed dataframe

    Args:
      data_df - pd.DataFrame
        as returned by sleep_analysis.data.read_data
      fly_col: (optional) name of fly column to display - str
        if None, displays activity averaged across flies
      zg_faithful: (optional) if True, actogram is faithful to make_zeitgeber_df
        if False, actogram is chronologically faithful and not sensitive to errors in the light column

    """
    _data_df = data_df.copy(deep=True)

    if zg_faithful:
        _zg_actogram(data_df, fly_col)
    else:
        _chrono_actogram(data_df, fly_col)

    assert_df_equals(data_df, _data_df)


def _zg_actogram(data_df, fly_col):
    """Actogram that is faithful to make_zeitgeber_df and will crap out if there is a malformed light column

    Args:
      data_df - pd.DataFrame
        as returned by sleep_analysis.data.read_data
      fly_col: name of fly column to display - str
        if None, displays activity averaged across flies
    """
    fly_cols = fly_col or [col for col in data_df.columns if col.startswith("fly")]

    zg_df = make_zeitgeber_df(
        data_df, include_incomplete=True, timedelta_workaround=True
    )

    start_time = data_df.index[0]

    days = np.unique(zg_df.index.get_level_values(0))

    fig, axes = plt.subplots(nrows=len(days), figsize=(20, 16), sharex=True)

    if fly_col is not None:
        y_max = zg_df[fly_col].max()
    else:
        y_max = zg_df[fly_cols].mean(axis=1).max()

    for day_idx in days:
        if fly_col is not None:
            axes[day_idx].plot(zg_df.loc[day_idx][fly_col])
        else:
            axes[day_idx].plot(zg_df.loc[day_idx][fly_cols].mean(axis=1))

        axes[day_idx].set_ylim(top=y_max * 1.1)
        axes[day_idx].set_yticks([])

        axes[day_idx].xaxis.set_major_locator(mdates.HourLocator(interval=2))
        axes[day_idx].xaxis.set_major_formatter(mdates.DateFormatter("%H"))

        axes[day_idx].set_ylabel(
            (start_time + timedelta(days=int(day_idx))).strftime("%m/%d")
        )

        axes[day_idx].plot(zg_df.loc[day_idx]["light"] * y_max, color="yellow")


def _chrono_actogram(data_df, fly_col):
    """Actogram that is chronologically faithful and literal

    Args:
      data_df - pd.DataFrame
        as returned by sleep_analysis.data.read_data
      fly_col: name of fly column to display - str
        if None, displays activity averaged across flies
    """
    fly_cols = fly_col or [col for col in data_df.columns if col.startswith("fly")]

    days = data_df.index[-1].day - data_df.index[0].day

    fig, axes = plt.subplots(nrows=days, figsize=(20, 16), sharex=False)

    if fly_col is not None:
        y_max = data_df[fly_col].max()
    else:
        y_max = data_df[fly_cols].mean(axis=1).max()

    start_time = datetime(
        data_df.index[0].year, data_df.index[0].month, data_df.index[0].day
    )
    end_time = start_time + timedelta(hours=24)

    for day_idx in range(days):
        if fly_col is not None:
            axes[day_idx].plot(data_df.loc[start_time:end_time, fly_col])
        else:
            axes[day_idx].plot(data_df.loc[start_time:end_time, fly_cols].mean(axis=1))

        axes[day_idx].set_ylim(top=y_max * 1.1)
        axes[day_idx].set_yticks([])

        axes[day_idx].xaxis.set_major_locator(mdates.HourLocator(interval=2))
        axes[day_idx].xaxis.set_major_formatter(mdates.DateFormatter("%H"))

        # axis unit is days since epoch
        axes[day_idx].set_xlim(
            left=(start_time - UNIX_EPOCH).days,
            right=(end_time - UNIX_EPOCH).days,
        )

        # x isn't shared, emulate it by disabling all but the last ticks
        x_ticks = axes[day_idx].get_xticks()
        axes[day_idx].set_xticks([])

        axes[day_idx].set_ylabel(start_time.strftime("%m/%d"))

        axes[day_idx].plot(
            data_df.loc[start_time:end_time, "light"] * y_max, color="yellow"
        )

        start_time += timedelta(hours=24)
        end_time += timedelta(hours=24)

    axes[-1].set_xticks(x_ticks)


def single_trial_zeitgeber_fig(
    zg_dfs,
    labels,
    show="sleep",
    days=None,
    start_time=None,
    end_time=None,
    fast=True,
    fig_size=[6.4, 4.8],
    **kwargs,
):
    """Simplified interface into zeitgeber_fig supporting only a single trial for each a genotype

    Args:
      zg_dfs: list of zeitgeber dataframes, one for each genotype - [pd.DataFrame]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
      labels: list of labels corresponding to dataframes, one for each genotype - [str]
      show: (optional) select whether to plot either sleep or activity - str, one of ['sleep', 'activity']
      days: (optional) day or days to average over. By default all days are used - [int] or int
        Can either select one day (with a single int), all days (by default) or a list of days
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      fast: (optional) if True don't plot confidence intervals  - bool
      kwargs: other kwargs are passed to zeitgeber_fig

    Returns:
      fig - matplotlib.figure.Figure
    """
    zg_dfs = [[zg_df] for zg_df in zg_dfs]

    if isinstance(days, list):
        days = [[days] for _ in range(len(labels))]

    return zeitgeber_fig(
        zg_dfs,
        labels,
        show=show,
        days=days,
        start_time=start_time,
        end_time=end_time,
        fast=fast,
        fig_size=fig_size,
        **kwargs,
    )


def zeitgeber_fig(
    zg_dfs,
    labels,
    show="sleep",
    days=None,
    start_time=None,
    end_time=None,
    fast=True,
    fig_size=[6.4, 4.8],
    colors=None,
    legend_placement="best",
    legend=True,
    **kwargs,
):
    """Figure level plot of sleep/activity supporting multiple trials for each genotype

    Averages over flies and days

    Only a limited amount of data validation is performed. Missing data/weird light cycles will lead to funky results

    Args:
      zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
      labels: list of labels corresponding to dataframes, one for each genotype - [str]
      show: (optional) select whether to plot either sleep or activity - str, one of ['sleep', 'activity']
      days: (optional) day or days to average over. By default all days are used - [[[int]]] or int
        Can either select one day (with a single int), all days (by default) or a selection from each trial like this
          [[genotype_1_trial_1_days, genotype_1_trial_2_days], [genotype_2_trial_1_days, genotype_2_trial_2_days]]
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      fast: (optional) if True don't plot confidence intervals  - bool
      fig_size: (optional) passed to matplotlib.pyplot.subplots - list
      colors: (optional) list of color labels passed to matplotlib.axes.Axes.plot() - list
      legend_placement: (optional) legend placement option as specified for matplotlib.pyplot.legend - str
      legend: (optional) if True, draw legend - bool
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()

    Returns:
      fig - matplotlib.figure.Figure
    """
    assert isinstance(zg_dfs, list)

    _zg_dfs = [[zg_df.copy(deep=True) for zg_df in gt_trials] for gt_trials in zg_dfs]

    assert isinstance(labels, list)
    assert len(zg_dfs) == len(labels)

    if colors is not None:
        assert len(colors) == len(zg_dfs)

    if isinstance(days, list):
        assert len(days) == len(labels)
        for trial_days in days:
            assert isinstance(trial_days, list)
    else:
        days = [days] * len(labels)

    for trials in zg_dfs:
        assert isinstance(trials, list)
        for zg_df in trials:
            assert isinstance(zg_df, pd.DataFrame)

            if not np.array_equal(
                zg_dfs[0][0].loc[0]["light"], zg_df.loc[0]["light"], equal_nan=True
            ):
                warn("Discrepancy in light column. Check data integrity")

    start_time = start_time or timedelta()
    end_time = end_time or timedelta(hours=24)

    fig, ax = plt.subplots(figsize=fig_size)

    # ==========================================================================
    #                              SLEEP AXIS
    # ==========================================================================

    for idx, (trials, trial_days, label) in enumerate(zip(zg_dfs, days, labels)):
        if colors is not None:
            zeitgeber_plot(
                trials,
                label=label,
                show=show,
                days=trial_days,
                start_time=start_time,
                end_time=end_time,
                confidence_intervals=not fast,
                axes=ax,
                color=colors[idx],
                **kwargs,
            )
        else:
            zeitgeber_plot(
                trials,
                label=label,
                show=show,
                days=trial_days,
                start_time=start_time,
                end_time=end_time,
                confidence_intervals=not fast,
                axes=ax,
                **kwargs,
            )

    xticks, xtick_labels = zip(*[(120 * idx, 2 * idx) for idx in range(13)])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.margins(0, 0.05)

    sns.move_legend(ax, legend_placement)

    if not legend:
        ax.legend().remove()

    # ==========================================================================
    #                             LIGHTBAR AXIS
    # ==========================================================================

    # create an axes beneath ax. The width of lb_ax will be 10%
    # of ax and the padding between lb_ax and ax will be fixed at 0.3 inch.
    divider = make_axes_locatable(ax)
    lb_ax = divider.append_axes("bottom", size="10%", pad=0.3)

    lb_ax.margins(0, 0)
    lb_ax.axis("off")

    lb_height = 0.01

    # three cases:
    #  1. time slices straddle light switch
    #  2. time slices lie within lights on
    #  3. time slices lie within lights off

    idx = pd.IndexSlice
    light_series = (
        zg_dfs[0][0].loc(axis=0)[idx[0, start_time:end_time]]["light"].to_numpy()
    )

    # handle nans
    if np.isnan(light_series).sum() > 0:
        nan_mask = np.where(np.isnan(light_series))[0]
        light_series[nan_mask] = light_series[nan_mask[-1] + 1]

    lights_on_bounds = contiguous_regions(light_series)

    assert len(lights_on_bounds) <= 1

    # cases 1 and 2
    if len(lights_on_bounds) == 1:
        # case 1
        if lights_on_bounds[0][1] < len(light_series):
            lb_ax.barh(
                0,
                width=lights_on_bounds[0][1],
                height=lb_height,
                left=0,
                facecolor="yellow",
            )

            lb_ax.barh(
                0,
                width=len(light_series) - lights_on_bounds[0][1],
                height=lb_height,
                left=lights_on_bounds[0][1],
                facecolor="black",
            )
        # case 2
        else:
            lb_ax.barh(
                0, width=len(light_series), height=lb_height, left=0, facecolor="yellow"
            )
    # case 3
    else:
        lb_ax.barh(
            0, width=len(light_series), height=lb_height, left=0, facecolor="black"
        )

    assert_data_equals(zg_dfs, _zg_dfs)

    return fig


# TODO: debug confidence intervals for multiple intervals
# TODO: debug shape error in stack for different num of fly columns
def zeitgeber_plot(
    zg_dfs,
    show="sleep",
    days=None,
    start_time=None,
    end_time=None,
    axes=None,
    label=None,
    confidence_intervals=True,
    **kwargs,
):
    """Lower level plot of sleep/activity over time. Averages over flies and days

    Args:
      zg_dfs: list of zeitgeber dfs, as returned by sleep_analysis.analysis.make_zeitgeber_df - [pd.DataFrame]
        Treated as multiple trials of the same genotype, data is combined.
      show: (optional) select whether to plot either sleep or activity - str, one of ['sleep', 'activity']
      days: (optional) day or days to average over. By default all days are used - [[int]] or int
        Can either select one day (with a single int), all days (by default), or a selection from each trial
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      axes: (optional) - matplotlib.axes.Axes
      label: (optional) - str
      confidence_intervals: (optional) if True use slower dataframe method - bool
        otherwise, just plot means
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()
    """
    assert isinstance(zg_dfs, list)
    assert show in ["sleep", "activity"]

    # validate days
    if isinstance(days, int):
        days = [[days]] * len(zg_dfs)

    if days is None:
        days = [list(np.unique(zg_df.index.get_level_values(0))) for zg_df in zg_dfs]

    assert isinstance(days, list)
    assert len(days) == len(zg_dfs)
    assert all([isinstance(day_idxs, list) for day_idxs in days])

    # validate times
    start_time = start_time or timedelta()
    end_time = end_time or timedelta(hours=24)

    assert isinstance(start_time, timedelta)
    assert isinstance(end_time, timedelta)
    assert start_time < end_time

    if confidence_intervals:
        melted_dfs = []

        for trial_idx, (day_idxs, zg_df) in enumerate(zip(days, zg_dfs)):
            _zg_df = zg_df.copy(deep=True)

            fly_cols = [col for col in _zg_df.columns if col.startswith("fly")]
            new_fly_cols = [f"{col}_{trial_idx}" for col in fly_cols]

            if show == "sleep":
                _zg_df = sleep_filter(_zg_df)

            _zg_df = _zg_df.rename(
                {col: new_col for col, new_col in zip(fly_cols, new_fly_cols)}, axis=1
            )

            # date selection
            idx = pd.IndexSlice
            _zg_df = _zg_df.loc(axis=0)[idx[day_idxs, start_time:end_time]]

            # pandas bullshit transformation from wide to long format
            # this takes all of the values of the fly columns and puts them in a single columns
            #  so that seaborn can aggregate over them
            melted_df = pd.melt(
                _zg_df.reset_index(), id_vars=["zg_time"], value_vars=new_fly_cols
            )

            # convert timedelta column to datetime column as workaround for dumb
            #  seaborn tick label issues
            melted_df["zg_time"] = melted_df["zg_time"].apply(
                lambda td: datetime(
                    year=10,
                    month=1,
                    day=1,
                    hour=td.seconds // 3600,
                    minute=(td.seconds // 60) % 60,
                )
            )

            melted_dfs.append(melted_df)

        ax = sns.lineplot(
            data=pd.concat(melted_dfs, ignore_index=True),
            x="zg_time",
            y="value",
            label=label,
            ax=axes,
            **kwargs,
        )

        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

    else:
        samples = []

        for day_idxs, zg_df in zip(days, zg_dfs):
            if show == "sleep":
                zg_df = sleep_filter(zg_df)

            for day in day_idxs:
                samples.append(
                    zg_df.loc[
                        (day, start_time):(day, end_time), zg_df.columns != "light"
                    ].to_numpy()
                )

        sample_tps = [len(sample) for sample in samples]

        if min(sample_tps) != max(sample_tps):
            warn(
                f"Mismatched sample lengths, discarding {max(sample_tps) - min(sample_tps)} timepoints"
            )

        samples = np.stack([sample[: min(sample_tps)] for sample in samples], axis=-1)

        sns.lineplot(data=np.mean(samples, axis=(1, 2)), label=label, ax=axes, **kwargs)


def per_diem_fig(
    zg_dfs,
    labels,
    start_time=None,
    end_time=None,
    start_day=None,
    end_day=None,
    fig_size=[6.4, 4.8],
    colors=None,
    legend_placement="best",
    legend=True,
    **kwargs,
):
    """Figure level plot of total sleep per diem

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
      fig_size: (optional) passed to matplotlib.pyplot.subplots - list
      colors: (optional) list of color labels passed to matplotlib.axes.Axes.plot() - list
      legend_placement: (optional) legend placement option as specified for matplotlib.pyplot.legend - str
      legend: (optional) if True, draw legend - bool
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()

    Returns:
      fig - matplotlib.figure.Figure
    """
    assert isinstance(zg_dfs, list)

    _zg_dfs = [[zg_df.copy(deep=True) for zg_df in gt_trials] for gt_trials in zg_dfs]

    assert isinstance(labels, list)
    assert len(zg_dfs) == len(labels)

    if colors is not None:
        assert len(colors) == len(zg_dfs)

    for trials in zg_dfs:
        assert isinstance(trials, list)
        for zg_df in trials:
            assert isinstance(zg_df, pd.DataFrame)

    fig, ax = plt.subplots(figsize=fig_size)

    for idx, (trials, label) in enumerate(zip(zg_dfs, labels)):
        if colors is not None:
            per_diem_plot(
                trials,
                label=label,
                start_time=start_time,
                end_time=end_time,
                start_day=start_day,
                end_day=end_day,
                axes=ax,
                color=colors[idx],
                **kwargs,
            )
        else:
            per_diem_plot(
                trials,
                label=label,
                start_time=start_time,
                end_time=end_time,
                start_day=start_day,
                end_day=end_day,
                axes=ax,
                **kwargs,
            )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    sns.move_legend(ax, legend_placement)

    if not legend:
        ax.legend().remove()

    assert_data_equals(zg_dfs, _zg_dfs)

    return fig


def per_diem_plot(
    zg_dfs,
    start_time=None,
    end_time=None,
    start_day=None,
    end_day=None,
    axes=None,
    label=None,
    **kwargs,
):
    """Lower level plot of total sleep per diem

    Args:
      zg_dfs: list of zeitgeber dfs, as returned by sleep_analysis.analysis.make_zeitgeber_df - pd.DataFrame
        Treated as multiple trials of the same genotype, data is combined.
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      start_day: (optional) by default all days are used - int
      end_day: (optional) by default all days are used - int
      axes: (optional) - matplotlib.axes.Axes
      label: (optional) - str
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()

    """
    # validate dfs
    assert isinstance(zg_dfs, list)
    for zg_df in zg_dfs:
        assert isinstance(zg_df, pd.DataFrame)

    melted_dfs = []

    for trial_idx, zg_df in enumerate(zg_dfs):
        fly_cols = [col for col in zg_df.columns if col.startswith("fly")]
        new_fly_cols = [f"{col}_{trial_idx}" for col in fly_cols]

        zg_df = zg_df.rename(
            {col: new_col for col, new_col in zip(fly_cols, new_fly_cols)}, axis=1
        )

        total_sleep_df = total_sleep(
            zg_df[new_fly_cols],
            start_time=start_time,
            end_time=end_time,
            start_day=start_day,
            end_day=end_day,
        )

        melted_df = pd.melt(
            total_sleep_df.reset_index(), id_vars=["day"], value_vars=new_fly_cols
        )

        melted_dfs.append(melted_df)

    ax = sns.lineplot(
        data=pd.concat(melted_dfs, ignore_index=True),
        x="day",
        y="value",
        label=label,
        ax=axes,
        **kwargs,
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.margins(0, 0.05)
    ax.set_ylabel("sleep (min)")


def cumulative_sleep_fig(
    zg_dfs,
    labels,
    fig_size=[6.4, 4.8],
    colors=None,
    legend_placement="best",
    legend=True,
    start_time=None,
    end_time=None,
    start_day=None,
    end_day=None,
    x_labels=None,
    **kwargs,
):
    """Figure level plot of average total sleep per fly per day

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
      labels: list of labels corresponding to dataframes, one for each genotype - [str]
      fig_size: (optional) passed to matplotlib.pyplot.subplots - list
      colors: (optional) list of color labels passed to sns.boxplot() - list
      legend_placement: (optional) legend placement option as specified for matplotlib.pyplot.legend - str
      legend: (optional) if True, draw legend - bool
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      start_day: (optional) by default all days are used - int
      end_day: (optional) by default all days are used - int
      x_labels: (optional) if passed, overrides light bars - [str]
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()

    Returns:
      fig - matplotlib.figure.Figure
    """
    return _cumulative_fig(
        zg_dfs,
        labels,
        "sleep",
        fig_size=fig_size,
        colors=colors,
        legend_placement=legend_placement,
        legend=legend,
        start_time=start_time,
        end_time=end_time,
        start_day=start_day,
        end_day=end_day,
        x_labels=x_labels,
        **kwargs,
    )


def cumulative_bout_number_fig(
    zg_dfs,
    labels,
    fig_size=[6.4, 4.8],
    colors=None,
    legend_placement="best",
    legend=True,
    start_time=None,
    end_time=None,
    start_day=None,
    end_day=None,
    x_labels=None,
    **kwargs,
):
    """Figure level plot of average number of bouts per fly per day

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
      labels: list of labels corresponding to dataframes, one for each genotype - [str]
      fig_size: (optional) passed to matplotlib.pyplot.subplots - list
      colors: (optional) list of color labels passed to sns.boxplot() - list
      legend_placement: (optional) legend placement option as specified for matplotlib.pyplot.legend - str
      legend: (optional) if True, draw legend - bool
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      start_day: (optional) by default all days are used - int
      end_day: (optional) by default all days are used - int
      x_labels: (optional) if passed, overrides light bars - [str]
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()

    Returns:
      fig - matplotlib.figure.Figure
    """
    return _cumulative_fig(
        zg_dfs,
        labels,
        "bout_num",
        fig_size=fig_size,
        colors=colors,
        legend_placement=legend_placement,
        legend=legend,
        start_time=start_time,
        end_time=end_time,
        start_day=start_day,
        end_day=end_day,
        x_labels=x_labels,
        **kwargs,
    )


def cumulative_bout_duration_fig(
    zg_dfs,
    labels,
    fig_size=[6.4, 4.8],
    colors=None,
    legend_placement="best",
    legend=True,
    start_time=None,
    end_time=None,
    start_day=None,
    end_day=None,
    x_labels=None,
    **kwargs,
):
    """Figure level plot of average duration of bouts per fly per day

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
      labels: list of labels corresponding to dataframes, one for each genotype - [str]
      fig_size: (optional) passed to matplotlib.pyplot.subplots - list
      colors: (optional) list of color labels passed to sns.boxplot() - list
      legend_placement: (optional) legend placement option as specified for matplotlib.pyplot.legend - str
      legend: (optional) if True, draw legend - bool
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      start_day: (optional) by default all days are used - int
      end_day: (optional) by default all days are used - int
      x_labels: (optional) if passed, overrides light bars - [str]
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()

    Returns:
      fig - matplotlib.figure.Figure
    """
    return _cumulative_fig(
        zg_dfs,
        labels,
        "bout_len",
        fig_size=fig_size,
        colors=colors,
        legend_placement=legend_placement,
        legend=legend,
        start_time=start_time,
        end_time=end_time,
        start_day=start_day,
        end_day=end_day,
        x_labels=x_labels,
        **kwargs,
    )


def cumulative_activity_fig(
    zg_dfs,
    labels,
    fig_size=[6.4, 4.8],
    colors=None,
    legend_placement="best",
    legend=True,
    start_time=None,
    end_time=None,
    start_day=None,
    end_day=None,
    x_labels=None,
    **kwargs,
):
    """Figure level plot of average activity per fly per day

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
      labels: list of labels corresponding to dataframes, one for each genotype - [str]
      fig_size: (optional) passed to matplotlib.pyplot.subplots - list
      colors: (optional) list of color labels passed to sns.boxplot() - list
      legend_placement: (optional) legend placement option as specified for matplotlib.pyplot.legend - str
      legend: (optional) if True, draw legend - bool
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      start_day: (optional) by default all days are used - int
      end_day: (optional) by default all days are used - int
      x_labels: (optional) if passed, overrides light bars - [str]
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()

    Returns:
      fig - matplotlib.figure.Figure
    """
    return _cumulative_fig(
        zg_dfs,
        labels,
        "activity",
        fig_size=fig_size,
        colors=colors,
        legend_placement=legend_placement,
        legend=legend,
        start_time=start_time,
        end_time=end_time,
        start_day=start_day,
        end_day=end_day,
        x_labels=x_labels,
        **kwargs,
    )


# TODO: validate no missing timepoints, particularly harmful for this plot
#    whether this is easy or annoying depends on whether I have to support fancy slicing and use zg_dfs
def _cumulative_fig(
    zg_dfs,
    labels,
    quantity,
    fig_size=[6.4, 4.8],
    colors=None,
    legend_placement="best",
    legend=True,
    start_time=None,
    end_time=None,
    start_day=None,
    end_day=None,
    x_labels=None,
    **kwargs,
):
    """Figure level plot for cumulative quantities

    Args:
     zg_dfs: list of list of zeitgeber dataframes - [[pd.DataFrame]]
        as returned by sleep_analysis.analysis.make_zeitgeber_df
        Supports multiple trials for each genotype like this
          [[genotype_1_trial_1, genotype_1_trial_2], [genotype_2_trial_1, genotype_2_trial_2]]
      labels: list of labels corresponding to dataframes, one for each genotype - [str]
      quantity: one of sleep, bout_num, bout_len - str
      fig_size: (optional) passed to matplotlib.pyplot.subplots - list
      colors: (optional) list of color labels passed to sns.boxplot() - list
      legend_placement: (optional) legend placement option as specified for matplotlib.pyplot.legend - str
      legend: (optional) if True, draw legend - bool
      start_time: (optional) zeitgeber start time - timedelta
      end_time: (optional) zeitgeber end time - timedelta
      start_day: (optional) by default all days are used - int
      end_day: (optional) by default all days are used - int
      x_labels: (optional) if passed, overrides light bars - [str]
      kwargs: other kwargs are passed to matplotlib.axes.Axes.plot()

    Returns:
      fig - matplotlib.figure.Figure
    """
    assert isinstance(zg_dfs, list)

    _zg_dfs = [[zg_df.copy(deep=True) for zg_df in gt_trials] for gt_trials in zg_dfs]

    assert isinstance(labels, list)
    assert len(zg_dfs) == len(labels)

    if colors is not None:
        assert len(colors) == len(labels)

    for trials in zg_dfs:
        assert isinstance(trials, list)
        for zg_df in trials:
            assert isinstance(zg_df, pd.DataFrame)

    assert timedelta(hours=0) <= start_time <= timedelta(hours=24)
    assert timedelta(hours=0) <= end_time <= timedelta(hours=24)

    zt0_straddled = False

    # handle case where start_time and_time straddle ZT0
    if not start_time < end_time:
        assert (start_time - timedelta(hours=24)) < timedelta(hours=0) < end_time

        # not supported due to ambiguity in day slicing when zt0 is straddled
        assert start_day is None
        assert end_day is None

        zt0_straddled = True

    fig, ax = plt.subplots(figsize=fig_size)

    # ==========================================================================
    #                              SLEEP AXIS
    # ==========================================================================

    if not zt0_straddled:
        grouped_df = group_by_light(
            zg_dfs,
            labels,
            quantity=quantity,
            start_day=start_day,
            end_day=end_day,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        grouped_df = pd.concat(
            [
                group_by_light(
                    zg_dfs,
                    labels,
                    quantity=quantity,
                    start_day=start_day,
                    end_day=end_day,
                    start_time=start_time,
                    end_time=timedelta(hours=24),
                ),
                group_by_light(
                    zg_dfs,
                    labels,
                    quantity=quantity,
                    start_day=start_day,
                    end_day=end_day,
                    start_time=timedelta(hours=0),
                    end_time=end_time,
                ),
            ]
        )

    if colors is not None:
        palette = {label: color for label, color in zip(labels, colors)}
    else:
        palette = None

    # if light and dark are both present in the light column, set order to light, dark
    # otherwise, set it to the element that is present
    order = sorted(grouped_df.loc[:, "light"].unique())[::-1]

    sns.boxplot(
        data=grouped_df,
        x="light",
        y=quantity,
        hue="label",
        ax=ax,
        palette=palette,
        order=order,
        **kwargs,
    )

    ax.set_xlabel("")

    if quantity == "sleep":
        ax.set_ylabel("Sleep (min)")
    if quantity == "bout_num":
        ax.set_ylabel("Number of bouts")
    if quantity == "bout_len":
        ax.set_ylabel("Bout duration (min)")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    sns.move_legend(ax, legend_placement)

    if not legend:
        ax.legend().remove()

    # ==========================================================================
    #                             LIGHTBAR AXIS
    # ==========================================================================

    if x_labels is None:
        if start_time is not None or end_time is not None:
            raise NotImplementedError()

        # create an axes beneath ax. The width of lb_ax will be 10%
        # of ax and the padding between lb_ax and ax will be fixed at 0.3 inch.
        divider = make_axes_locatable(ax)
        lb_ax = divider.append_axes("bottom", size="10%", pad=0.1)

        lb_ax.margins(0, 0)
        lb_ax.axis("off")

        lb_height = 0.01

        lb_ax.barh(0, width=1, height=lb_height, left=-0.5, facecolor="yellow")

        lb_ax.barh(0, width=1, height=lb_height, left=0.5, facecolor="black")

        ax.get_xaxis().set_visible(False)
    else:
        ax.set_xticklabels(x_labels)

    assert_data_equals(zg_dfs, _zg_dfs)

    return fig
