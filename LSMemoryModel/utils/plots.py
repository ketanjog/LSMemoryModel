"""
Only LTM suffixed functions will work.

Data is unavailable for the other functions. 
"""

import os
from LSMemoryModel.constants.paths import OBJECT_DUMP_PATH, FIGURES_PATH
import sys
import numpy as np
from LSMemoryModel.utils.save import load_object
import matplotlib.pyplot as plt
from scipy.stats import norm

# Time
TIME = 100000

# Values swept over:
sweep = {
    "stm_learning_rate": [0.5],
    "ltm_learning_rate": [0.0, 0.01, 0.1],
    "num_contexts": [10],
    "num_actions": [100, 500, 1000, 5000],
    "context_epsilon": [0.01],
    "action_epsilon_positive": [0.01],
    "action_epsilon_negative": [0.01],
    "beta_stm": [10, 50, 100],
    "beta_ltm_params": [10, 50, 100],
}

# Short hand for title
name = {
    "stm_learning_rate": "s_LR: ",
    "ltm_learning_rate": "l_LR",
    "num_contexts": "#C: ",
    "num_actions": "#A: ",
    "context_epsilon": "eps: ",
    "action_epsilon_positive": "a_ep_p",
    "action_epsilon_negative": "a_ep_n",
    "beta_stm": "beta_s",
    "beta_ltm": "beta_l",
}

# we write down 15 colors
colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
    "#aec7e8",  # light blue
    "#ffbb78",  # orange
    "#98df8a",  # yellow-green
    "#ff9896",  # pink
    "#c5b0d5",  # light purple
    "#c49c94",  # chestnut brown
    "#f7b6d2",  # light pink
    "#c7c7c7",  # light gray
    "#dbdb8d",  # tan
    "#9edae5",  # blue-green
]


def make_plots_pretty():
    """
    Sets the standard plotting parameters
    """
    plt.style.use("ggplot")
    font = {"size": 22}
    plt.rc("font", **font)


# Add variables here...functionality stays the same.
def load_set(
    stm_learning_rate=None,
    num_contexts=None,
    num_actions=None,
    context_epsilon=None,
    beta=None,
):
    # check if at least one argument is left unspecified
    if all([i is not None for i in locals().values()]):
        print(
            "One argument needs to be left unspecified, so we can graph along all its values"
        )
        return 0

    # Check if exactly 1 argument is left unspecified
    elif sum(x is None for x in locals().values()) > 1:
        print(
            "More than one argument is None, we currently don't have heatmap functionality"
        )
        return 0

    else:
        # Make title
        title_vars = locals().items()
        title = ""
        for key, value in title_vars:
            if value is not None:
                label = name[key]
                title += str(label) + str(value) + " "

        data = {}
        # Store a copy of input args
        params = list(locals().values())[:5]

        # Get the name of the argument that is unspecified, and iterate over its swept values
        iterable = list(locals().keys())[list(locals().values()).index(None)]
        for param in sweep[iterable]:
            # Construct filename and load data from it
            file_name = (
                "_".join([str(param) if _ is None else str(_) for _ in params]) + ".pkl"
            )
            data[file_name] = load_object(file_name)

        return data, iterable, title


def plot_ci(
    stm_learning_rate=None,
    num_contexts=None,
    num_actions=None,
    context_epsilon=None,
    beta=None,
    save=False,
    folder=None,
):
    """
    Makes a plot of the lines with confidence intervals.
    Assumes that runs is a dictionary with they keys being the names of the runs
    and the values being a list of length num_runs. You can pass an $x_axis$
    value to change the values displayed on the x axis.
    """
    # Set basic params
    make_plots_pretty()

    # Get data, x_label and title
    runs, x_label, title = load_set(
        stm_learning_rate, num_contexts, num_actions, context_epsilon, beta
    )
    # Set Y_label, name and other figure constants
    y_label = "Reward/Time"
    name = title
    ci = (0.95,)
    figsize = (10, 10)
    x_axis = np.arange(0, len(runs.items()) + 1, 1)
    horizontal_line_width = 0.25

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    run_colors = {}
    c = 0

    # Iterate through items, create plots
    for rname, run in runs.items():
        if rname not in run_colors:
            run_colors[rname] = colors[c]
            c += 1

        run = np.array(run)
        run = run + np.random.normal(0, 0.01, run.shape)

        # get the confidence interval
        mean, std = np.mean(run, axis=0), np.std(run, axis=0)
        conf_int = norm.interval(ci, loc=mean, scale=std)

        # Set the interval location
        left = x_axis[c] - horizontal_line_width / 2
        top = conf_int[1]
        right = x_axis[c] + horizontal_line_width / 2
        bottom = conf_int[0]

        # plot the interval
        ax.plot([x_axis[c], x_axis[c]], [top, bottom], color=run_colors[rname])
        ax.plot([left, right], [top, top], color=run_colors[rname])
        ax.plot([left, right], [bottom, bottom], color=run_colors[rname])
        ax.plot(x_axis[c], mean, "o", color=run_colors[rname], label=rname)

    plt.xticks([1, 2, 3, 4], sweep[x_label])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # fig.legend(loc="upper right", prop={"size": 16})

    if save:
        if not os.path.exists(os.path.join(FIGURES_PATH, folder)):
            os.makedirs(os.path.join(FIGURES_PATH, folder))
        fig.savefig(os.path.join(FIGURES_PATH, folder, f"{name}-ci.png"), dpi=300)


def plot_ci_ltm_run(
    stm_learning_rate=None,
    ltm_learning_rate=None,
    num_contexts=None,
    num_actions=None,
    context_epsilon=None,
    action_epsilon_positive=None,
    action_epsilon_negative=None,
    beta_stm=None,
    beta_ltm=None,
    save=False,
    folder=None,
):
    """
    Makes a plot of the lines with confidence intervals.
    Assumes that runs is a dictionary with they keys being the names of the runs
    and the values being a list of length num_runs. You can pass an $x_axis$
    value to change the values displayed on the x axis.
    """
    # Set basic params
    make_plots_pretty()

    # Get data, x_label and title
    runs, x_label, title = load_set_ltm(
        stm_learning_rate=stm_learning_rate,
        ltm_learning_rate=ltm_learning_rate,
        num_contexts=num_contexts,
        num_actions=num_actions,
        context_epsilon=context_epsilon,
        action_epsilon_positive=action_epsilon_positive,
        action_epsilon_negative=action_epsilon_negative,
        beta_stm=beta_stm,
        beta_ltm=beta_ltm,
    )
    # Set Y_label, name and other figure constants
    y_label = "Total Rewards"
    name = title
    ci = (0.95,)
    figsize = (10, 10)
    x_axis = np.arange(0, len(runs.items()) + 1, 1)
    horizontal_line_width = 0.25

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    run_colors = {}
    c = 0

    # Iterate through items, create plots
    for rname, run in runs.items():
        if rname not in run_colors:
            run_colors[rname] = colors[c]
            c += 1

        run = np.array(run)
        run = run + np.random.normal(0, 0.01, run.shape)

        # get the confidence interval
        mean, std = np.mean(run / TIME, axis=0), np.std(run / TIME, axis=0)
        conf_int = norm.interval(ci, loc=mean, scale=std)

        # Set the interval location
        left = x_axis[c] - horizontal_line_width / 2
        top = conf_int[1]
        right = x_axis[c] + horizontal_line_width / 2
        bottom = conf_int[0]

        # plot the interval
        ax.plot([x_axis[c], x_axis[c]], [top, bottom], color=run_colors[rname])
        ax.plot([left, right], [top, top], color=run_colors[rname])
        ax.plot([left, right], [bottom, bottom], color=run_colors[rname])
        ax.plot(x_axis[c], mean, "o", color=run_colors[rname], label=rname)

    plt.xticks(np.arange(1, len(sweep[x_label]) + 1, 1), sweep[x_label])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # fig.legend(loc="upper right", prop={"size": 16})

    if save:
        if not os.path.exists(os.path.join(FIGURES_PATH, folder)):
            os.makedirs(os.path.join(FIGURES_PATH, folder))
        fig.savefig(os.path.join(FIGURES_PATH, folder, f"{name}-ci.png"), dpi=300)


def load_set_ltm(
    stm_learning_rate=None,
    ltm_learning_rate=None,
    num_contexts=None,
    num_actions=None,
    context_epsilon=None,
    action_epsilon_positive=None,
    action_epsilon_negative=None,
    beta_stm=None,
    beta_ltm=None,
):

    title_vars_full = locals().copy()
    title_vars = title_vars_full.values()

    # check if at least one argument is left unspecified
    if all([i is not None for i in locals().values()]):
        print(
            "One argument needs to be left unspecified, so we can graph along all its values"
        )
        return 0

    # Check if exactly 1 argument is left unspecified
    elif sum(x is None for x in locals().values()) > 1:
        print(
            "More than one argument is None, we currently don't have heatmap functionality"
        )
        return 0

    else:
        # Make title

        title = ""

        for key, value in title_vars_full.items():
            if value is not None:
                label = name[key]
                title += str(label) + str(value) + " "

        data = {}
        # Store a copy of input args
        # CHANGE THIS number to be length of input args sequence
        params = list(title_vars)[:9]
        # print(params)
        # SO JANK PLS FIX WTF
        j = params.pop()
        params.insert(0, j)

        # if ltm_learning_rate is not None and ltm_learning_rate not in [
        #     stm_learning_rate * i for i in sweep["ltm_learning_rate"]
        # ]:
        #     print("invalid LTM learning rate param")
        #     return 0

        # Get the name of the argument that is unspecified, and iterate over its swept values
        iterable = list(title_vars_full.keys())[list(locals().values()).index(None)]

        sweep_vals = []
        if iterable == "ltm_learning_rate":
            sweep_vals = [stm_learning_rate * i for i in sweep["ltm_learning_rate"]]
        else:
            sweep_vals = sweep[iterable]

        for param in sweep_vals:
            # Construct filename and load data from it
            file_name = (
                "_".join([str(param) if _ is None else str(_) for _ in params]) + ".pkl"
            )
            data[file_name] = load_object(file_name)

        return data, iterable, title
