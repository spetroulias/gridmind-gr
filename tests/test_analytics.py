from datetime import date
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from gridmind.main import app
from gridmind.services import analytics
from gridmind.models import load_forecast, res_forecast
from gridmind.data.database import get_database_url


@pytest.mark.parametrize('question,dataset,technology', [
    ('Load 2026-01-15', 'load', None),
    ('RES 2026-01-15', 'res', None),
    ('Renewable production 2026-01-15', 'res', None),
    ('Generation by source 2026-01-15', 'generation', None),
    ('Gas production 2026-01-15', 'generation', 'natural_gas'),
    ('Forecast load 2026-01-15', 'forecast', None),
    ('Κατανάλωση 15 Ιανουαρίου 2026', 'load', None),
])
def test_intents(question, dataset, technology):
    assert analytics.parse_request(question) == (dataset, date(2026, 1, 15), date(2026, 1, 15), technology)


def test_date_range():
    assert analytics.parse_request('load 2026-01-01 to 2026-01-31')[1:3] == (date(2026, 1, 1), date(2026, 1, 31))


def test_api_health_without_optional_services():
    client = TestClient(app)
    assert client.get('/health').status_code == 200
    assert client.post('/api/v1/chat', json={'message': ' '}).status_code == 422
    assert client.post('/api/v1/chat', json={'message': 'load 2026-02-31'}).status_code == 422
    assert client.get('/api/v1/data/load').status_code == 422
    assert client.get('/api/v1/forecasts/load?date=invalid').status_code == 422


def test_chat_and_history_return_chart_data(monkeypatch):
    frame = pd.DataFrame({'timestamp': pd.date_range('2026-01-15', periods=24, freq='h'), 'load_mwh': [5000.0] * 24})
    monkeypatch.setattr(analytics.queries, 'get_historical_load', lambda *args: frame)
    client = TestClient(app)
    chat = client.post('/api/v1/chat', json={'message': 'load 2026-01-15'}).json()
    direct = client.get('/api/v1/data/load?date=2026-01-15').json()
    assert chat['data'] == direct['data']
    assert len(chat['data']) == 24
    assert '120,000.00' in chat['answer']


def test_generation_aggregates_system_peak(monkeypatch):
    frame = pd.DataFrame({'timestamp': pd.to_datetime(['2026-01-15', '2026-01-15']), 'technology': ['hydro', 'res'], 'production_mwh': [100., 200.]})
    monkeypatch.setattr(analytics.queries, 'get_hourly_generation_mix', lambda *args: frame)
    result = analytics.get_analytics('generation', date(2026, 1, 15), date(2026, 1, 15))
    assert 'peak: 300.00' in result['answer']


def test_database_driver_and_password(monkeypatch):
    monkeypatch.setenv('POSTGRES_PASSWORD', 'contains@:/#')
    url = get_database_url()
    assert url.drivername == 'postgresql+psycopg'
    assert url.password == 'contains@:/#'


@pytest.mark.parametrize('kind', ['load', 'res'])
def test_lags_use_elapsed_hours_with_missing_observations(kind):
    times = pd.date_range('2026-01-01', periods=300, freq='h').delete(200)
    history = pd.DataFrame({'timestamp': times, f'{kind}_mwh': range(1000, 1000 + len(times))})
    result = load_forecast.load_model_data(history) if kind == 'load' else res_forecast.build_model_data(history)
    # Timestamp 224 lacks its exact 24-hour lag. Row-based shift would incorrectly retain it.
    assert pd.Timestamp('2026-01-10 08:00') not in set(result.timestamp)


def test_recursive_forecast_without_future_actuals():
    class YesterdayModel:
        def predict(self, features):
            return features['load_lag_24'].to_numpy()
    history = pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=24 * 8, freq='h'), 'load_mwh': [5000.] * (24 * 8)})
    result = load_forecast.make_future_forecast(YesterdayModel(), history, '2026-01-10')
    assert len(result) == 24
    assert result.forecast_mwh.eq(5000).all()
    assert result.actual_mwh.isna().all()
    assert result.forecast_type.eq('recursive_future').all()


def test_historical_forecast_retrains_before_target(monkeypatch):
    history = pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=24 * 10, freq='h'), 'load_mwh': [5000.] * 240})
    trained = []
    def train(frame):
        trained.append(frame.timestamp.max())
        return 'cutoff model'
    monkeypatch.setattr(load_forecast, 'train_model', train)
    monkeypatch.setattr(load_forecast, 'make_historical_forecast', lambda **kwargs: kwargs['model'])
    assert load_forecast.forecast_date('all-history model', history, '2026-01-09') == 'cutoff model'
    assert trained == [pd.Timestamp('2026-01-08 23:00')]


def test_no_data_response(monkeypatch):
    monkeypatch.setattr(analytics.queries, 'get_historical_res', lambda *args: pd.DataFrame())
    result = analytics.get_analytics('res', date(2026, 1, 15), date(2026, 1, 15))
    assert result['data'] == []
    assert result['answer'].startswith('No res data')
