from datetime import date
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from gridmind.main import app
from gridmind.services.conversation import ChatContext, resolve_request
from gridmind.services import chatbot


def context(dataset='load', start='2026-01-15', end=None, technology=None):
    return ChatContext(dataset=dataset, start_date=start, end_date=end or start, technology=technology)


@pytest.mark.parametrize('message,expected', [
    ('and the next day?', ('load', date(2026, 1, 16), date(2026, 1, 16), None)),
    ('and the previous day?', ('load', date(2026, 1, 14), date(2026, 1, 14), None)),
    ('show renewables instead', ('res', date(2026, 1, 15), date(2026, 1, 15), None)),
    ('what about natural gas?', ('generation', date(2026, 1, 15), date(2026, 1, 15), 'natural_gas')),
    ('what was the peak?', ('load', date(2026, 1, 15), date(2026, 1, 15), None)),
    ('what was the minimum?', ('load', date(2026, 1, 15), date(2026, 1, 15), None)),
    ('και την επομενη ημερα;', ('load', date(2026, 1, 16), date(2026, 1, 16), None)),
    ('τον προηγουμενο μηνα', ('load', date(2025, 12, 15), date(2025, 12, 15), None)),
])
def test_followups(message, expected):
    assert resolve_request(message, context()) == expected


def test_forecast_and_filter_survive_date_only_followup():
    assert resolve_request('and 2027-09-13?', context('forecast', '2027-09-12')) == ('forecast', date(2027, 9, 13), date(2027, 9, 13), None)
    assert resolve_request('and the next day?', context('generation', technology='hydro'))[-1] == 'hydro'


def test_explicit_request_resets_filter_and_dates():
    previous = context('generation', technology='hydro')
    assert resolve_request('Show generation 2026-02-01', previous) == ('generation', date(2026, 2, 1), date(2026, 2, 1), None)
    assert resolve_request('Show RES 2026-02-01', previous)[0] == 'res'


def test_shift_range_and_leap_date():
    assert resolve_request('next month', context(start='2026-01-01', end='2026-01-31'))[1:3] == (date(2026, 2, 1), date(2026, 2, 28))
    assert resolve_request('same day next year', context('forecast', '2028-02-29'))[1:3] == (date(2029, 2, 28), date(2029, 2, 28))


def test_no_context_and_unknown_followups_do_not_invent_requests():
    with pytest.raises(ValueError, match='starting point'):
        resolve_request('and the next day?')
    assert resolve_request('why is it like that?', context()) is None
    assert resolve_request('why was load higher?', context()) is None
    with pytest.raises(ValueError, match='comparisons'):
        resolve_request('compare with next day', context())


@pytest.fixture
def api(monkeypatch):
    def analytics(dataset, start, end, technology=None):
        return {'answer': f'{dataset}: {start} to {end}', 'sources': [], 'data': [], 'dataset': dataset}
    monkeypatch.setattr(chatbot, 'get_analytics', analytics)
    return TestClient(app)


def test_api_chain_and_session_isolation(api):
    first = api.post('/api/v1/chat', json={'message':'hydro production 2026-01-15'}).json()
    second = api.post('/api/v1/chat', json={'message':'and the next day?', 'context': first['context']}).json()
    assert second['context']['start_date'] == '2026-01-16'
    assert second['context']['technology'] == 'hydro'
    third = api.post('/api/v1/chat', json={'message':'show RES instead', 'context':second['context']}).json()
    assert third['context']['dataset'] == 'res'
    assert third['context']['technology'] is None
    assert third['context']['start_date'] == '2026-01-16'
    # The shared service never stores another caller's context.
    assert api.post('/api/v1/chat', json={'message':'and the next day?'}).status_code == 422
    unrelated = api.post('/api/v1/chat', json={'message':'What can you do?', 'context':third['context']}).json()
    assert unrelated['context'] == third['context']


def test_invalid_client_context_rejected(api):
    bad = context().model_dump(mode='json')
    bad['dataset'] = 'arbitrary_table'
    assert api.post('/api/v1/chat', json={'message':'next day', 'context':bad}).status_code == 422


def test_streamlit_sends_context_and_new_chat_clears_it(monkeypatch, api):
    from streamlit.testing.v1 import AppTest
    import requests
    payloads = []
    class Response:
        ok = True
        def __init__(self, data): self.data = data
        def json(self): return self.data
    def request(method, url, **kwargs):
        payloads.append(kwargs['json'])
        response = api.post('/api/v1/chat', json=kwargs['json'])
        assert response.status_code == 200
        return Response(response.json())
    monkeypatch.setattr(requests, 'request', request)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'streamlit_app.py').run()
    app.chat_input[0].set_value('Load 2026-01-15').run()
    app.chat_input[0].set_value('and the next day?').run()
    assert not app.exception
    assert payloads[0]['context'] is None
    assert payloads[1]['context']['start_date'] == '2026-01-15'
    assert app.session_state['chat_context']['start_date'] == '2026-01-16'
    app.button[0].click().run()
    assert app.session_state['messages'] == []
    assert 'chat_context' not in app.session_state
