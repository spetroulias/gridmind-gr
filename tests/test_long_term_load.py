from datetime import date
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from gridmind.main import app
from gridmind.models import long_term_load as model
from gridmind.models import load_forecast
from gridmind.services.analytics import parse_request


@pytest.fixture
def history():
    times = pd.date_range('2023-01-01', '2025-12-31 23:00', freq='h')
    return pd.DataFrame({'timestamp': times, 'load_mwh': 5000. + times.hour.to_numpy() * 10})


def test_calendar_features_need_no_future_load():
    times = pd.date_range('2027-09-12', periods=24, freq='h')
    features = model.calendar_features(times)
    assert features.shape == (24, 5)
    assert not any('lag' in column for column in features)
    assert np.isfinite(features).all().all()


def test_year_ahead_projection_has_independent_validation_and_ordered_bounds(monkeypatch, history):
    fitted = []
    class CalendarModel:
        def fit(self, features, values):
            fitted.append(len(values))
            return self
        def predict(self, features):
            return 5100. + features.hour.to_numpy() * 10
    monkeypatch.setattr(model, 'build_model', CalendarModel)
    frame, metadata = model.make_long_term_forecast(history, '2026-12-31')
    assert len(frame) == 24
    assert frame.timestamp.min() == pd.Timestamp('2026-12-31')
    assert frame.timestamp.max() == pd.Timestamp('2026-12-31 23:00')
    assert fitted == [len(history) - 365 * 24, len(history)]
    assert metadata['validation_start'] == '2025-01-01 00:00:00'
    assert metadata['validation_metrics']['mae_mwh'] == 100
    assert metadata['error_band_mwh'] == 100
    assert frame.lower_mwh.le(frame.forecast_mwh).all()
    assert frame.upper_mwh.ge(frame.forecast_mwh).all()
    assert frame.lower_mwh.ge(0).all()


@pytest.mark.parametrize('target', ['2025-12-31', '2028-01-01'])
def test_invalid_target_rejected(history, target):
    with pytest.raises(ValueError):
        model.make_long_term_forecast(history, target)


def test_requires_two_complete_seasons(history):
    with pytest.raises(ValueError, match='two years'):
        model.make_long_term_forecast(history.tail(365 * 24), '2026-12-31')


def test_rejects_missing_season(history):
    history.loc[history.timestamp.dt.month == 7, 'load_mwh'] = np.nan
    with pytest.raises(ValueError, match='coverage'):
        model.make_long_term_forecast(history, '2026-12-31')


def test_year_ahead_chat_and_direct_api_use_long_term_model(monkeypatch, history):
    monkeypatch.setattr(load_forecast, 'load_history', lambda: history)
    def recursive_forbidden(*args, **kwargs):
        raise AssertionError('A long-term request must not run the recursive model')
    monkeypatch.setattr(load_forecast, 'train_model', recursive_forbidden)
    def long_term(history, target):
        frame = pd.DataFrame({'timestamp': pd.date_range(target, periods=24, freq='h'), 'forecast_mwh': [5000.] * 24, 'lower_mwh': [4000.] * 24, 'upper_mwh': [6000.] * 24})
        return frame, {'validation_metrics': {'mae_mwh': 400., 'wape_pct': 8.}}
    monkeypatch.setattr(model, 'make_long_term_forecast', long_term)
    client = TestClient(app)
    chat = client.post('/api/v1/chat', json={'message': 'Forecast load 2026-12-31'})
    direct = client.get('/api/v1/forecasts/load?date=2026-12-31')
    assert chat.status_code == direct.status_code == 200
    assert chat.json()['data'] == direct.json()['data']
    assert 'Long-term seasonal projection' in chat.json()['answer']
    assert chat.json()['forecast_metadata']['validation_metrics']['wape_pct'] == 8.


def test_relative_year_phrase():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo('Europe/Athens')).date()
    expected = (pd.Timestamp(today) + pd.DateOffset(years=1)).date()
    assert parse_request('Forecast load in one year') == ('forecast', expected, expected, None)
