import os
import yaml
from brane_packages.visualization.main import plot_distribution_wrapper, bar_chart_compare_wrapper, feature_group_bar_chart_wrapper
from brane_packages.processing.main import train_and_predict_wrapper, transform_fields_wrapper, drop_unuseful_columns_wrapper
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

