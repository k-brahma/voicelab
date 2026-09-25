"""E（OpenAI 構成）の「リクエストの組み立て」「費用の換算」「TTS の送受信」「1 往復の組み立て」。

**ネットワークを使わない。** OpenAI は ``httpx.MockTransport`` で、Gemini はダミーで差し替える。
OpenAI のクレジットも Gemini のトークンも 1 も使わずに、B と同じ流れに乗っていることを確かめる。
"""

import json

import httpx
import pytest

from voicelab import metrics, openai_path, tts_path, tts_provider
from voicelab.custom_path import AudioSink, Turn

SCENARIO = {
    'id': 'deploy-check',
    'text': 'デプロイの前に確認することを教えて',
    'expected_note': 'デプロイ手順',
}

#: 残高切れのときに OpenAI が返す本文（message が長く、code は後ろにある）。
QUOTA_BODY = {
    'error': {
        'message': (
            'You exceeded your current quota, please check your plan and billing details. '
            'For more information on this error, read the docs: '
            'https://platform.openai.com/docs/guides/error-codes/api-errors.'
        ),
        'type': 'insufficient_quota',
        'param': None,
        'code': 'insufficient_quota',
    }
}


class FakeChunk:
    """Gemini のストリームの断片のふり（B のテストと同じ形）。"""

    def __init__(self, text=None):
        self.text = text
        self.usage_metadata = None


def _mock_client(handler):
    """OpenAI のふりをする ``httpx.Client`` を作る関数を返す（鍵は受け取って捨てる）。"""

    def factory(_api_key):
        return httpx.Client(transport=httpx.MockTransport(handler))

    return factory


def _speech_handler(requests, *, audio=b'\x01\x02\x03\x04', status=200, body=None):
    """``/v1/models/<モデル>`` には 200、``/v1/audio/speech`` には決めた音（か失敗）を返す。"""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith('/v1/models/'):
            return httpx.Response(200, json={'id': 'gpt-4o-mini-tts', 'object': 'model'})
        if status >= 400:
            return httpx.Response(status, json=body or QUOTA_BODY)
        return httpx.Response(200, content=audio)

    return handler


# --------------------------------------------------------------------------- リクエストの組み立て


def test_speech_url_と温めの_url():
    assert openai_path.speech_url() == 'https://api.openai.com/v1/audio/speech'
    assert openai_path.warmup_url('gpt-4o-mini-tts') == 'https://api.openai.com/v1/models/gpt-4o-mini-tts'


def test_speech_body_はモデルと声と_pcm_を渡し末尾の空白を足さない():
    assert openai_path.speech_body('一文目。', model='gpt-4o-mini-tts', voice='marin') == {
        'model': 'gpt-4o-mini-tts',
        'input': '一文目。',
        'voice': 'marin',
        'response_format': 'pcm',
    }


def test_既定のモデルと声():
    assert openai_path.DEFAULT_OPENAI_TTS_MODEL == 'gpt-4o-mini-tts'
    assert openai_path.DEFAULT_OPENAI_TTS_VOICE == 'marin'
    assert openai_path.voice_of({}) == 'marin'
    assert openai_path.voice_of({'OPENAI_TTS_VOICE': 'coral'}) == 'coral'


def test_pcm_は_24kHz_で変換しない():
    assert openai_path.SPEC.sample_rate == 24000
    assert openai_path.SPEC.output_format == 'pcm_24000'


def test_認証は_Bearer_方式のヘッダで渡す():
    client = openai_path._make_client('abc')
    try:
        assert client.headers['Authorization'] == 'Bearer abc'
    finally:
        client.close()


# --------------------------------------------------------------------------- 費用


def test_費用は音の長さかける_1_分あたりの単価():
    one_minute = Turn(question='q', hits=[], pcm=b'\x00' * (24000 * 2 * 60))
    assert openai_path.audio_seconds(one_minute.pcm) == pytest.approx(60.0)
    assert openai_path.estimate_usd(one_minute) == pytest.approx(0.015)
    ten_seconds = Turn(question='q', hits=[], pcm=b'\x00' * (24000 * 2 * 10))
    assert openai_path.estimate_usd(ten_seconds) == pytest.approx(0.0025)
    assert openai_path.estimate_usd(Turn(question='q', hits=[])) == 0


def test_note_は_B_の_note_にドルを足す():
    turn = Turn(question='q', hits=[], sentences=['あ。'], reply='あ。', sent_chars=2,
                pcm=b'\x00' * (24000 * 2 * 60))
    note = openai_path.build_openai_note(SCENARIO, turn)
    assert '検索1位=None' in note
    assert note.endswith('$0.01500')


# --------------------------------------------------------------------------- TTS


def test_openai_tts_は文ごとに_1_本ずつ投げて音を溜める():
    requests: list[httpx.Request] = []
    sink = AudioSink()
    tts = openai_path.OpenAiTts(
        api_key='k', model='gpt-4o-mini-tts', voice='marin', sink=sink,
        client_factory=_mock_client(_speech_handler(requests)),
    )
    with tts:
        tts.send('一文目。')
        tts.send('二文目。')
        tts.finish()
        assert tts.wait(5) is True

    # 最初の 1 本は接続を温める GET（t0 より前）。そのあと文の数だけ POST
    assert [r.method for r in requests] == ['GET', 'POST', 'POST']
    assert requests[0].url.path == '/v1/models/gpt-4o-mini-tts'
    assert all(r.url.path == '/v1/audio/speech' for r in requests[1:])
    bodies = [json.loads(r.content) for r in requests[1:]]
    assert [b['input'] for b in bodies] == ['一文目。', '二文目。']
    assert all(b['voice'] == 'marin' and b['response_format'] == 'pcm' for b in bodies)
    assert sink.pcm() == b'\x01\x02\x03\x04' * 2
    assert tts.sent_sentences == 2
    assert tts.sent_chars == len('一文目。') * 2
    assert len(tts.ttfb_ms) == 2
    assert tts.error is None


def test_openai_tts_は残高切れを_error_に残して止まる():
    requests: list[httpx.Request] = []
    sink = AudioSink()
    tts = openai_path.OpenAiTts(
        api_key='k', model='m', voice='marin', sink=sink,
        client_factory=_mock_client(_speech_handler(requests, status=429)),
    )
    with tts:
        tts.send('一文目。')
        tts.send('二文目。')
        tts.finish()
        assert tts.wait(5) is True

    assert tts.error.startswith('HTTP 429 insufficient_quota')
    assert openai_path.is_payment_required(tts.error)
    assert sink.pcm() == b''
    assert len(requests) == 2  # 2 文目は投げない


def test_is_payment_required_は残高切れだけを拾い速度制限は拾わない():
    assert openai_path.is_payment_required('HTTP 429 insufficient_quota: You exceeded ...')
    assert openai_path.is_payment_required('HTTP 402: {}')
    assert not openai_path.is_payment_required('HTTP 429 rate_limit_exceeded: Rate limit ...')
    assert not openai_path.is_payment_required('HTTP 400 invalid_value: voice')
    assert not openai_path.is_payment_required(None)


def test_describe_http_error_は_code_を先に取り出す():
    body = json.dumps(QUOTA_BODY).encode()
    assert openai_path.describe_http_error(429, body).startswith('HTTP 429 insufficient_quota: You exceeded')
    assert openai_path.describe_http_error(500, b'oops') == 'HTTP 500: oops'


def test_openai_tts_鍵が通らなければ開く時点で止まる():
    def handler(request):
        return httpx.Response(401, json={'error': {'code': 'invalid_api_key'}})

    tts = openai_path.OpenAiTts(
        api_key='k', model='m', voice='marin', sink=AudioSink(), client_factory=_mock_client(handler)
    )
    with pytest.raises(openai_path.OpenAiPathError, match='401'):
        tts.open()


# --------------------------------------------------------------------------- 1 往復


def _patch_pipeline(monkeypatch, handler, saved, made=None):
    from voicelab import custom_path

    monkeypatch.setattr(
        custom_path, 'stream_gemini',
        lambda *a, **k: [FakeChunk('デプロイ手順によると、確認が要ります。')],
    )
    original = openai_path.OpenAiTts

    def make_tts(**kwargs):
        if made is not None:
            made.append(kwargs)
        return original(client_factory=_mock_client(handler), **kwargs)

    monkeypatch.setattr(openai_path, 'OpenAiTts', make_tts)
    # 保存は共通ランナー（tts_path）が行うので、差し替えもそちらに入れる
    monkeypatch.setattr(
        tts_path, 'save_audio_file',
        lambda *a, **k: saved.setdefault('audio', (k['label'], k['sample_rate'])),
    )
    monkeypatch.setattr(
        tts_path, 'save_transcript', lambda *a, **k: saved.setdefault('text', (k['label'], a[0]))
    )


def test_run_scenario_は_B_と同じ_Run_を返して_openai_の名前で残す(monkeypatch):
    saved: dict = {}
    made: list[dict] = []
    _patch_pipeline(monkeypatch, _speech_handler([]), saved, made)

    run = openai_path.run_scenario(
        SCENARIO, env={'OPENAI_API_KEY': 'k', 'GEMINI_API_KEY': 'g', 'OPENAI_TTS_VOICE': 'coral'}
    )

    assert run.path == openai_path.PATH_OPENAI == 'openai'
    assert run.credits == 0  # ElevenLabs のクレジットは使わない
    assert run.model == 'gpt-4o-mini-tts_coral'  # 置き場は <モデル>_<声>
    assert run.correct is None
    assert run.first_audio_ms >= 0 and run.reply_done_ms >= run.first_audio_ms
    assert '検索1位=デプロイ手順' in run.note
    assert '$' in run.note
    assert made[0]['voice'] == 'coral'  # .env の声が TTS まで届く
    assert saved['audio'] == ('openai', 24000)  # 24kHz のまま WAV にする
    label, text = saved['text']
    assert label == 'openai'
    assert '構成: E（検索 → gemini-3.6-flash → OpenAI gpt-4o-mini-tts）' in text
    assert 'ElevenLabs のクレジットは 0' in text
    assert '## OpenAI の声' in text and '- coral' in text
    assert '## OpenAI の文ごとの応答' in text


def test_run_scenario_は残高切れなら記録せずに止まる(monkeypatch):
    saved: dict = {}
    _patch_pipeline(monkeypatch, _speech_handler([], status=429), saved)

    with pytest.raises(openai_path.OpenAiPathError, match='insufficient_quota'):
        openai_path.run_scenario(SCENARIO, env={'OPENAI_API_KEY': 'k', 'GEMINI_API_KEY': 'g'})
    assert saved == {}


def test_run_scenario_は鍵が無ければ接続前に止まる():
    from voicelab.config import ConfigError

    with pytest.raises(ConfigError, match='OPENAI_API_KEY'):
        openai_path.run_scenario(SCENARIO, env={'GEMINI_API_KEY': 'g'})


def test_dry_run_は接続せず_B_と同じ検索結果を見せる():
    text = openai_path.describe_dry_run(SCENARIO, env={})
    assert 'デプロイ手順' in text
    assert 'gpt-4o-mini-tts' in text
    assert 'pcm_24000' in text
    assert '声 marin' in text
    assert '未設定' in text


# --------------------------------------------------------------------------- 登録と表


def test_登録簿に_openai_が載る():
    provider = tts_provider.get('openai')
    assert provider.label == 'B: 自前構成（OpenAI）'
    assert provider.required_env == ('OPENAI_API_KEY', 'GEMINI_API_KEY')
    assert 'API から読めない' in provider.describe_balance({})
    assert 'openai' in metrics.known_paths()


def test_report_に_OpenAI_の行が出る():
    run = metrics.Run(
        scenario_id='deploy-check', path='openai',
        first_audio_ms=900, reply_done_ms=2000, credits=0,
    )
    assert '| B: 自前構成（OpenAI） | 1 |' in metrics.render_report([run])
