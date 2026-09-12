import pandas as pd
import pytest
from gridmind.data.parsers import parse_system_load, parse_res, parse_generation


@pytest.fixture
def load_sheet(tmp_path):
    frame = pd.DataFrame(index=range(9), columns=range(26))
    frame.iloc[1] = ['Date', *range(1, 26)]
    frame.iloc[2] = ['15-01-2026', *([5375.0] * 24), 0.0]
    frame.iloc[7] = ['Date', *range(1, 26)]
    frame.iloc[8] = ['15-01-2026', *([-50.0] * 24), 0.0]
    path = tmp_path / 'load.xlsx'
    frame.to_excel(path, header=False, index=False)
    return path


def test_parse_system_load(load_sheet):
    df = parse_system_load(load_sheet)
    assert df.shape == (25, 4)
    assert df.iloc[0].net_load_mwh == 5375
    assert df.iloc[0].crete_flow_mwh == -50
    assert df.iloc[-1].period == 25


def test_parse_res(load_sheet):
    assert parse_res(load_sheet).res_mwh.sum() == 5375 * 24


def test_generation_excludes_subtotals(tmp_path):
    path = tmp_path / '20260115_generation.xlsx'
    frame = pd.DataFrame([
        [None, 'ΛΙΓΝΙΤΙΚΕΣ ΜΟΝΑΔΕΣ', 1, 2, 'SUM'],
        [None, 'Unit A', 10, 20, 30],
        [None, 'TOTAL LIGNITE', 10, 20, 30],
        [None, 'ΑΝΑΝΕΩΣΙΜΑ', 1, 2, 'SUM'],
        [None, 'PV', 0, 5, 5],
        [None, 'TOTAL RES', 0, 5, 5],
    ])
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name='System_Production', header=False, index=False)
    result = parse_generation(path)
    assert len(result) == 4
    assert result.production_mwh.sum() == 35
    assert set(result.technology) == {'lignite', 'res'}
