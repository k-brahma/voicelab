"""D（Deepgram 構成）の「リクエストの組み立て」「費用の換算」「TTS の送受信」「1 往復の組み立て」。

**ネットワークを使わない。** Deepgram は ``httpx.MockTransport`` で、Gemini はダミーで差し替える。
Deepgram の残高も Gemini のトークンも 1 も使わずに、B と同じ流れに乗っていることを確かめる。
"""

import json

import httpx
import pytest

from voicelab import credits, deepgram_path, metrics, tts_path
from voicelab.custom_path import AudioSink

SCENARIO = {
    'id': 'deploy-check',
    'text': 'デプロイの前に確認することを教えて',
    'expected_note': 'デプロイ手順',
}


class FakeChunk:
    """Gemini のストリームの断片のふり（B のテストと同じ形）。"""

    def __init__(self, text=None):
        self.text = text
        self.usage_metadata = None


def _mock_client(handler):
    """Deepgram のふりをする ``httpx.Client`` を作る関数を返す（鍵は受け取って捨てる）。"""

    def factory(_api_key):
        return httpx.Client(transport=httpx.MockTransport(handler))

    return factory


def _speak_handler(requests, *, audio=b'\x01\x02\x03\x04', status=200):
    """``/v1/models`` には 200、``/v1/speak`` には決めた音（か失敗）を返す。"""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'tts': []})
        if status >= 400:
            return httpx.Response(status, json={'err_code': 'ASR_PAYMENT_REQUIRED'})
        return httpx.Response(200, content=audio)

    return handler


# --------------------------------------------------------------------------- リクエストの組み立て


def test_speak_url_にモデルと_B_と同じ出力形式が入る():
    url = deepgram_path.speak_url('aura-2-izanami-ja')
    assert url.startswith('https://api.deepgram.com/v1/speak?')
    assert 'model=aura-2-izanami-ja' in url
    assert 'encoding=linear16' in url
    assert 'sample_rate=16000' in url  # B の pcm_16000 と同じ
    assert 'container=none' in url  # WAV のヘッダを付けない（文ごとにつなぐため）


def test_speak_body_は本文だけで末尾の空白を足さない():
    assert deepgram_path.speak_body('一文目。') == {'text': '一文目。'}


def test_既定のモデルは日本語の_aura_2():
    from voicelab.config import DEFAULT_DEEPGRAM_TTS_MODEL

    assert DEFAULT_DEEPGRAM_TTS_MODEL.startswith('aura-2-')
    assert DEFAULT_DEEPGRAM_TTS_MODEL.endswith('-ja')


# --------------------------------------------------------------------------- 費用


def test_費用は文字数かける千文字あたりの単価():
    assert deepgram_path.estimate_usd(1000) == pytest.approx(0.030)
    assert deepgram_path.estimate_usd(100) == pytest.approx(0.003)
    assert deepgram_path.estimate_usd(0) == 0


def test_note_は_B_の_note_にドルを足す():
    from voicelab.custom_path import Turn

    turn = Turn(question='q', hits=[], sentences=['あ。'], reply='あ。', sent_chars=100)
    note = deepgram_path.build_deepgram_note(SCENARIO, turn)
    assert '検索1位=None' in note
    assert note.endswith('$0.00300')


# --------------------------------------------------------------------------- TTS


def test_deepgram_tts_は文ごとに_1_本ずつ投げて音を溜める():
    requests: list[httpx.Request] = []
    sink = AudioSink()
    tts = deepgram_path.DeepgramTts(
        api_key='k', model='aura-2-izanami-ja', sink=sink,
        client_factory=_mock_client(_speak_handler(requests)),
    )
    with tts:
        tts.send('一文目。')
        tts.send('二文目。')
        tts.finish()
        assert tts.wait(5) is True

    # 最初の 1 本は接続を温める GET（t0 より前）。そのあと文の数だけ POST
    assert [r.method for r in requests] == ['GET', 'POST', 'POST']
    bodies = [json.loads(r.content) for r in requests[1:]]
    assert bodies == [{'text': '一文目。'}, {'text': '二文目。'}]
    assert sink.pcm() == b'\x01\x02\x03\x04' * 2
    assert tts.sent_sentences == 2
    assert tts.sent_chars == len('一文目。') * 2
    assert len(tts.ttfb_ms) == 2
    assert tts.error is None


def test_deepgram_tts_は_402_を_error_に残して止まる():
    requests: list[httpx.Request] = []
    sink = AudioSink()
    tts = deepgram_path.DeepgramTts(
        api_key='k', model='m', sink=sink,
        client_factory=_mock_client(_speak_handler(requests, status=402)),
    )
    with tts:
        tts.send('一文目。')
        tts.send('二文目。')
        tts.finish()
        assert tts.wait(5) is True

    assert tts.error.startswith('HTTP 402')
    assert deepgram_path.is_payment_required(tts.error)
    assert sink.pcm() == b''
    assert len(requests) == 2  # 2 文目は投げない


def test_is_payment_required_は_402_だけを拾う():
    assert deepgram_path.is_payment_required('HTTP 402: {}')
    assert not deepgram_path.is_payment_required('HTTP 400: {}')
    assert not deepgram_path.is_payment_required(None)


def test_deepgram_tts_鍵が通らなければ開く時点で止まる():
    def handler(request):
        return httpx.Response(401, json={'err_code': 'INVALID_AUTH'})

    tts = deepgram_path.DeepgramTts(
        api_key='k', model='m', sink=AudioSink(), client_factory=_mock_client(handler)
    )
    with pytest.raises(deepgram_path.DeepgramPathError, match='401'):
        tts.open()


# --------------------------------------------------------------------------- 1 往復


def _patch_pipeline(monkeypatch, handler, saved):
    from voicelab import custom_path

    monkeypatch.setattr(
        custom_path, 'stream_gemini',
        lambda *a, **k: [FakeChunk('デプロイ手順によると、確認が要ります。')],
    )
    original = deepgram_path.DeepgramTts

    def make_tts(**kwargs):
        return original(client_factory=_mock_client(handler), **kwargs)

    monkeypatch.setattr(deepgram_path, 'DeepgramTts', make_tts)
    # 保存は共通ランナー（tts_path）が行うので、差し替えもそちらに入れる
    monkeypatch.setattr(
        tts_path, 'save_audio_file', lambda *a, **k: saved.setdefault('audio', k['label'])
    )
    monkeypatch.setattr(
        tts_path, 'save_transcript', lambda *a, **k: saved.setdefault('text', (k['label'], a[0]))
    )


def test_run_scenario_は_B_と同じ_Run_を返して_deepgram_の名前で残す(monkeypatch):
    saved: dict = {}
    _patch_pipeline(monkeypatch, _speak_handler([]), saved)

    run = deepgram_path.run_scenario(
        SCENARIO, env={'DEEPGRAM_API_KEY': 'k', 'GEMINI_API_KEY': 'g'}
    )

    assert run.path == metrics.PATH_DEEPGRAM == 'deepgram'
    assert run.credits == 0  # ElevenLabs のクレジットは使わない
    assert run.correct is None
    assert run.first_audio_ms >= 0 and run.reply_done_ms >= run.first_audio_ms
    assert '検索1位=デプロイ手順' in run.note
    assert '$' in run.note
    assert saved['audio'] == 'deepgram'
    label, text = saved['text']
    assert label == 'deepgram'
    assert '構成: D（検索 → gemini-3.6-flash → Deepgram aura-2-izanami-ja）' in text
    assert 'ElevenLabs のクレジットは 0' in text
    assert '## Deepgram の文ごとの応答' in text


def test_run_scenario_は_402_なら記録せずに止まる(monkeypatch):
    saved: dict = {}
    _patch_pipeline(monkeypatch, _speak_handler([], status=402), saved)

    with pytest.raises(deepgram_path.DeepgramPathError, match='402'):
        deepgram_path.run_scenario(SCENARIO, env={'DEEPGRAM_API_KEY': 'k', 'GEMINI_API_KEY': 'g'})
    assert saved == {}


def test_run_scenario_は鍵が無ければ接続前に止まる():
    from voicelab.config import ConfigError

    with pytest.raises(ConfigError, match='DEEPGRAM_API_KEY'):
        deepgram_path.run_scenario(SCENARIO, env={'GEMINI_API_KEY': 'g'})


def test_dry_run_は接続せず_B_と同じ検索結果を見せる():
    text = deepgram_path.describe_dry_run(SCENARIO, env={})
    assert 'デプロイ手順' in text
    assert 'aura-2-izanami-ja' in text
    assert 'linear16_16000' in text
    assert '未設定' in text


# --------------------------------------------------------------------------- 残高と表


def test_残高の空配列は残高なし():
    balance = credits.parse_deepgram_balances({'balances': []})
    assert balance.entries == 0
    assert balance.amount == 0
    assert '残高なし' in balance.describe()


def test_残高は全件を足す():
    balance = credits.parse_deepgram_balances(
        {'balances': [{'amount': 150.5, 'units': 'usd'}, {'amount': 49.5, 'units': 'usd'}]}
    )
    assert balance.amount == pytest.approx(200.0)
    assert '200.0000 usd' in balance.describe()


def test_残高の形が違えば_CreditsError():
    with pytest.raises(credits.CreditsError):
        credits.parse_deepgram_balances({})


def test_report_に_D_の行が出る():
    run = metrics.Run(
        scenario_id='deploy-check', path=metrics.PATH_DEEPGRAM,
        first_audio_ms=900, reply_done_ms=2000, credits=0,
    )
    assert '| B: 自前構成（Deepgram） | 1 |' in metrics.render_report([run])


def test_認証は_Token_方式のヘッダで渡す():
    client = deepgram_path._make_client('abc')
    try:
        assert client.headers['Authorization'] == 'Token abc'
    finally:
        client.close()
