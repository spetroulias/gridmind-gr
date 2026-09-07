import pandas as pd
import pytest

from gridmind.data.validation import validate_system_load


def test_valid_system_load_passes():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-15")],
            "period": [1],
            "net_load_mwh": [5000.0],
            "crete_flow_mwh": [-50.0],
        }
    )

    validate_system_load(df)


def test_invalid_period_fails():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-15")],
            "period": [30],
            "net_load_mwh": [5000.0],
            "crete_flow_mwh": [-50.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Invalid system load period",
    ):
        validate_system_load(df)