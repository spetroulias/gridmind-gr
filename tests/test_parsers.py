from gridmind.data.parsers import parse_system_load


def test_parse_system_load():
    df = parse_system_load(
        "data/raw/20260115_RealTimeSCADASystemLoad_01.xls"
    )

    assert df.shape == (25, 4)

    assert list(df.columns) == [
        "date",
        "period",
        "net_load_mwh",
        "crete_flow_mwh",
    ]

    assert df.iloc[0]["period"] == 1
    assert df.iloc[-1]["period"] == 25

    assert df.iloc[0]["net_load_mwh"] == 5375.0
    assert df.iloc[0]["crete_flow_mwh"] == -50.0