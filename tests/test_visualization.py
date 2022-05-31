import os
import yaml
from brane_packages.visualization.main import plot_distribution_wrapper, bar_chart_compare_wrapper, feature_group_bar_chart_wrapper
import pytest

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'train.csv')


def test_plot_distribution_wrapper():
    os.environ["DATA"] = DATA_PATH
    os.environ["FEATURE"] = "Age"
    os.environ["PLOT_TITLE"] = "Age Distribution of Passengers"
    try:
        yaml_result = plot_distribution_wrapper()
        test_result = yaml.safe_load(yaml_result)
        assert(
            "output" in test_result and
            isinstance(test_result["output"], str) and
            test_result["output"].endswith('.png')
        )
    except Exception as e:
        assert False


def test_bar_chart_compare_wrapper():
    os.environ["DATA"] = DATA_PATH
    os.environ["FEATURE_Y"] = "Survived"
    os.environ["FEATURE_1"] = "Pclass"
    os.environ["FEATURE_2"] = "Sex"
    os.environ["Y_LABEL"] = "Survival Rate"
    os.environ["PLOT_TITLE"] = "Survival rate by sex and class"
    try:
        yaml_result = bar_chart_compare_wrapper()
        test_result = yaml.safe_load(yaml_result)
        assert(
            "output" in test_result and
            isinstance(test_result["output"], str) and
            test_result["output"].endswith('.png')
        )
    except Exception as e:
        assert False


def test_feature_group_bar_chart_wrapper():
    os.environ["DATA"] = DATA_PATH
    os.environ["FEATURE_Y_BINARY"] = "Survived"
    os.environ["FEATURE_Y_INDEX"] = "2"
    os.environ["FEATURE_Y_INDEX0"] = "Passenger Survived"
    os.environ["FEATURE_Y_INDEX1"] = "Passenger Died"
    os.environ["FEATURE"] = "Pclass"
    os.environ["Y_LABEL"] = "# of passengers"
    os.environ["PLOT_TITLE"] = "# of passengers surviving by class"
    try:
        yaml_result = feature_group_bar_chart_wrapper()
        test_result = yaml.safe_load(yaml_result)
        assert(
            "output" in test_result and
            isinstance(test_result["output"], str) and
            test_result["output"].endswith('.png')
        )
    except Exception as e:
        assert False

