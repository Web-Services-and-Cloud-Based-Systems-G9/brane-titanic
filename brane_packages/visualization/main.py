#!/usr/bin/env python3
import os
import sys
import yaml
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
plt.style.use("seaborn-whitegrid")


def read_dataset(data_fn: str) -> pd.DataFrame:
    data = pd.read_csv(data_fn)
    return data


def generate_figure_name() -> str:
    figure_name = str(uuid.uuid4()) + ".png"
    return figure_name


def plot_distribution(data: str, feature: str, plot_title: str) -> str:
    data = read_dataset(data)
    bins = 30
    hist = True
    f_size = (7, 7)
    fig, ax = plt.subplots(figsize=f_size)
    ax.set_title(plot_title)
    sns.distplot(data[feature], color="g", bins=bins, ax=ax)
    file_name = generate_figure_name()
    plt.savefig(file_name)
    return file_name


def bar_chart_compare(data: str, feature_y: str, feature_1: str, feature_2: str, y_label: str, plot_title: str) -> str:
    data = read_dataset(data)
    f_size = (7, 7)
    plt.figure(figsize=f_size)
    plt.title(plot_title)
    if feature_2 == "None":
        feature_2 = None
    sns.barplot(x=feature_1, y=feature_y, hue=feature_2, ci=None, data=data).set_ylabel(y_label)
    file_name = generate_figure_name()
    plt.savefig(file_name)
    return file_name


def feature_group_bar_chart(data: str, feature_y_binary: str, feature_y_index: List[str], feature: str, y_label: str, plot_title: str) -> str:
    data = read_dataset(data)
    f_size = (7, 7)
    survived = data[data[feature_y_binary] == 1][feature].value_counts()
    dead = data[data[feature_y_binary] == 0][feature].value_counts()
    df_survived_dead = pd.DataFrame([survived, dead])
    df_survived_dead.index = feature_y_index
    ax = df_survived_dead.plot(kind="bar", stacked=True, figsize=f_size, rot=0)
    ax.set_ylabel(y_label)
    file_name = generate_figure_name()
    plt.title(plot_title)
    plt.savefig(file_name)
    return file_name


def plot_distribution_wrapper():
    arg_data = os.environ["DATA"]
    arg_feature = os.environ["FEATURE"]
    arg_plot_title = os.environ["PLOT_TITLE"]
    output = plot_distribution(arg_data, arg_feature, arg_plot_title)
    yaml_result = yaml.dump({"output": output})
    print(yaml_result)
    return yaml_result


def bar_chart_compare_wrapper():
    arg_data = os.environ["DATA"]
    arg_feature_y = os.environ["FEATURE_Y"]
    arg_feature_1 = os.environ["FEATURE_1"]
    arg_feature_2 = os.environ["FEATURE_2"]
    arg_y_label = os.environ["Y_LABEL"]
    arg_plot_title = os.environ["PLOT_TITLE"]
    output = bar_chart_compare(arg_data, arg_feature_y, arg_feature_1, arg_feature_2, arg_y_label, arg_plot_title)
    yaml_result = yaml.dump({"output": output})
    print(yaml_result)
    return yaml_result


def feature_group_bar_chart_wrapper():
    arg_data = os.environ["DATA"]
    arg_feature_y_binary = os.environ["FEATURE_Y_BINARY"]
    arg_feature = os.environ["FEATURE"]
    arg_y_label = os.environ["Y_LABEL"]
    arg_plot_title = os.environ["PLOT_TITLE"]
    arg_feature_y_index = [
        os.environ[f"FEATURE_Y_INDEX{i}"] for i in range(int(os.environ["FEATURE_Y_INDEX"]))
    ]
    output = feature_group_bar_chart(arg_data, arg_feature_y_binary, arg_feature_y_index, arg_feature, arg_y_label,
                                     arg_plot_title)
    yaml_result = yaml.dump({"output": output})
    print(yaml_result)
    return yaml_result


if __name__ == "__main__":
    command = sys.argv[1]

    if command == "plot_distribution":
        plot_distribution_wrapper()

    elif command == "bar_chart_compare":
        bar_chart_compare_wrapper()

    elif command == "feature_group_bar_chart":
        feature_group_bar_chart_wrapper()
