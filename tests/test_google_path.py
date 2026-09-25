"""F（Google 構成）の「リクエストの組み立て」「WAV ヘッダの落とし方」「費用の換算」「TTS の送受信」
「1 往復の組み立て」。

**ネットワークを使わない。** Google は ``httpx.MockTransport`` で、Gemini はダミーで差し替える。
Google の文字数も Gemini のトークンも 1 も使わずに、B と同じ流れに乗っていることを確かめる。
"""

import base64
import json
import struct

import httpx
import pytest

from voicelab import google_path, metrics, tts_path
from voicelab.custom_path import AudioSink

SCENARIO = {
    'id': 'deploy-check',
    'text': 'デプロイの前に確認することを教えて',
    'expected_note': 'デプロイ手順',
}

VOICE = 'ja-JP-Chirp3-HD-Kore'


class FakeChunk:
    """Gemini のストリームの断片のふり（B のテストと同じ形）。"""

    def __init__(self, text=None):
        self.text = text
        self.usage_metadata = None


def _wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Google の ``LINEAR16`` と同じ 44 バイトのヘッダを付けた WAV（2026-09-25 の実物と同じ並び）。"""
    fmt = struct.pack('<HHIIHH', 1, 1, sample_rate, sample_rate * 2, 2, 16)
    return (
        b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVE'
        + b'fmt ' + struct.pack('<I', 16) + fmt
        + b'data' + struct.pack('<I', len(pcm)) + pcm
    )


def _mock_client(handler):
    """Google のふりをする ``httpx.Client`` を作る関数を返す（鍵は受け取って捨てる）。"""

    def factory(_api_key):
        return httpx.Client(transport=httpx.MockTransport(handler))

    return factory


def _synth_handler(requests, *, pcm=b'\x01\x02\x03\x04', status=200, error=None):
    """``/v1/voices`` には 200、``/v1/text:synthesize`` には決めた音（か失敗）を返す。"""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == '/v1/voices':
            return httpx.Response(200, json={'voices': []})
        if status >= 400:
            return httpx.Response(status, json={'error': error or {'code': status, 'message': 'x'}})
        return httpx.Response(200, json={'audioContent': base64.b64encode(_wav(pcm)).decode()})

    return handler


BILLING_ERROR = {
    'code': 403,
    'message': 'This API method requires billing to be enabled. Please enable billing on project #1.',
    'status': 'PERMISSION_DENIED',
}
QUOTA_ERROR = {
    'code': 429,
    'message': 'Quota exceeded for quota metric ...',
    'status': 'RESOURCE_EXHAUSTED',
}


# --------------------------------------------------------------------------- リクエストの組み立て


def test_synthesize_url_と温める_url():
    assert google_path.synthesize_url() == 'https://texttospeech.googleapis.com/v1/text:synthesize'
    assert google_path.voices_url() == 'https://texttospeech.googleapis.com/v1/voices?languageCode=ja-JP'


def test_synthesize_body_に声と_B_と同じ出力形式が入る():
    assert google_path.synthesize_body('一文目。', VOICE) == {
        'input': {'text': '一文目。'},  # 末尾の空白は足さない
        'voice': {'languageCode': 'ja-JP', 'name': VOICE},
        'audioConfig': {'audioEncoding': 'LINEAR16', 'sampleRateHertz': 16000},  # B の pcm_16000 と同じ
    }


def test_既定の声は日本語の_chirp_3_hd():
    assert google_path.DEFAULT_GOOGLE_TTS_VOICE.startswith('ja-JP-Chirp3-HD-')
    assert google_path.SPEC.default_model == google_path.DEFAULT_GOOGLE_TTS_VOICE
    assert google_path.SPEC.model({'GOOGLE_TTS_VOICE': 'ja-JP-Chirp3-HD-Leda'}) == 'ja-JP-Chirp3-HD-Leda'


def test_認証は_x_goog_api_key_ヘッダで渡す():
    client = google_path._make_client('abc')
    try:
        assert client.headers['x-goog-api-key'] == 'abc'
        assert 'Authorization' not in client.headers
    finally:
        client.close()


# --------------------------------------------------------------------------- WAV ヘッダ


def test_wav_ヘッダの_44_バイトを落とす():
    pcm = b'\x10\x20' * 50
    wav = _wav(pcm)
    assert wav[:4] == b'RIFF' and len(wav) == 44 + len(pcm)
    assert google_path.strip_wav_header(wav) == pcm


def test_riff_で始まらなければそのまま返す():
    assert google_path.strip_wav_header(b'\x01\x02\x03\x04') == b'\x01\x02\x03\x04'
    assert google_path.strip_wav_header(b'') == b''


def test_data_の前に別のチャンクがあっても_data_を探す():
    pcm = b'\x05\x06' * 4
    wav = _wav(pcm)
    extra = b'LIST' + struct.pack('<I', 3) + b'abc' + b'\x00'  # 奇数長は 1 バイト詰める
    wav = wav[:36] + extra + wav[36:]
    assert google_path.strip_wav_header(wav) == pcm


def test_decode_audio_は_base64_を戻してヘッダを落とす():
    pcm = b'\x01\x02\x03\x04'
    payload = {'audioContent': base64.b64encode(_wav(pcm)).decode()}
    assert google_path.decode_audio(payload) == pcm


# --------------------------------------------------------------------------- 費用


def test_費用は文字数かける千文字あたりの単価():
    assert google_path.estimate_usd(1000) == pytest.approx(0.030)
    assert google_path.estimate_usd(100) == pytest.approx(0.003)
    assert google_path.estimate_usd(0) == 0


def test_note_は_B_の_note_にドルを足す():
    from voicelab.custom_path import Turn

    turn = Turn(question='q', hits=[], sentences=['あ。'], reply='あ。', sent_chars=100)
    note = google_path.build_google_note(SCENARIO, turn)
    assert '検索1位=None' in note
    assert note.endswith('$0.00300')


def test_残高は読めない旨を_1_行で返す():
    line = google_path.describe_balance({})
    assert line.startswith('Google: ')
    assert '100 万文字' in line


# --------------------------------------------------------------------------- TTS


def test_google_tts_は文ごとに_1_本ずつ投げてヘッダを落とした音を溜める():
    requests: list[httpx.Request] = []
    sink = AudioSink()
    tts = google_path.GoogleTts(
        api_key='k', voice=VOICE, sink=sink,
        client_factory=_mock_client(_synth_handler(requests)),
    )
    with tts:
        tts.send('一文目。')
        tts.send('二文目。')
        tts.finish()
        assert tts.wait(5) is True

    # 最初の 1 本は接続を温める GET（t0 より前）。そのあと文の数だけ POST
    assert [r.method for r in requests] == ['GET', 'POST', 'POST']
    assert requests[0].url.path == '/v1/voices'
    bodies = [json.loads(r.content) for r in requests[1:]]
    assert [b['input']['text'] for b in bodies] == ['一文目。', '二文目。']
    assert all(b['voice']['name'] == VOICE for b in bodies)
    # 1 文ぶんが 1 回の add() で入り、つないでもヘッダ（RIFF）が混ざらない
    assert sink.chunks == [b'\x01\x02\x03\x04', b'\x01\x02\x03\x04']
    assert b'RIFF' not in sink.pcm()
    assert tts.sent_sentences == 2
    assert tts.sent_chars == len('一文目。') * 2
    assert len(tts.ttfb_ms) == 2
    assert tts.error is None


def test_google_tts_は課金切れの_403_を_error_に残して止まる():
    requests: list[httpx.Request] = []
    sink = AudioSink()
    tts = google_path.GoogleTts(
        api_key='k', voice=VOICE, sink=sink,
        client_factory=_mock_client(_synth_handler(requests, status=403, error=BILLING_ERROR)),
    )
    with tts:
        tts.send('一文目。')
        tts.send('二文目。')
        tts.finish()
        assert tts.wait(5) is True

    assert tts.error.startswith('HTTP 403 PERMISSION_DENIED: This API method requires billing')
    assert google_path.is_payment_required(tts.error)
    assert sink.pcm() == b''
    assert len(requests) == 2  # 2 文目は投げない


def test_google_tts_は応答の形が違えば_error_に残す():
    def handler(request):
        if request.url.path == '/v1/voices':
            return httpx.Response(200, json={'voices': []})
        return httpx.Response(200, json={'unexpected': True})

    sink = AudioSink()
    tts = google_path.GoogleTts(api_key='k', voice=VOICE, sink=sink, client_factory=_mock_client(handler))
    with tts:
        tts.send('一文目。')
        tts.finish()
        assert tts.wait(5) is True
    assert '応答の形が想定と違います' in tts.error
    assert sink.pcm() == b''


def test_is_payment_required_は課金と割り当てだけを拾う():
    assert google_path.is_payment_required('HTTP 403 PERMISSION_DENIED: ... requires billing to be enabled')
    assert google_path.is_payment_required('HTTP 403 PERMISSION_DENIED: Quota exceeded')
    assert google_path.is_payment_required('HTTP 429 RESOURCE_EXHAUSTED: Quota exceeded')
    assert not google_path.is_payment_required('HTTP 403 PERMISSION_DENIED: API has not been used')
    assert not google_path.is_payment_required('HTTP 400 INVALID_ARGUMENT: voice')
    assert not google_path.is_payment_required(None)


def test_describe_http_error_は_message_を取り出す():
    response = httpx.Response(429, json={'error': QUOTA_ERROR})
    assert google_path.describe_http_error(response) == 'HTTP 429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric ...'
    assert google_path.describe_http_error(httpx.Response(502, text='bad gateway')) == 'HTTP 502: bad gateway'


def test_google_tts_鍵が通らなければ開く時点で止まる():
    def handler(request):
        return httpx.Response(
            400, json={'error': {'code': 400, 'message': 'API key not valid.', 'status': 'INVALID_ARGUMENT'}}
        )

    tts = google_path.GoogleTts(api_key='k', voice=VOICE, sink=AudioSink(), client_factory=_mock_client(handler))
    with pytest.raises(google_path.GooglePathError, match='400 INVALID_ARGUMENT'):
        tts.open()


# --------------------------------------------------------------------------- 1 往復


def _patch_pipeline(monkeypatch, handler, saved):
    from voicelab import custom_path

    monkeypatch.setattr(
        custom_path, 'stream_gemini',
        lambda *a, **k: [FakeChunk('デプロイ手順によると、確認が要ります。')],
    )
    original = google_path.GoogleTts

    def make_tts(**kwargs):
        return original(client_factory=_mock_client(handler), **kwargs)

    monkeypatch.setattr(google_path, 'GoogleTts', make_tts)
    # 保存は共通ランナー（tts_path）が行うので、差し替えもそちらに入れる
    monkeypatch.setattr(
        tts_path, 'save_audio_file',
        lambda *a, **k: saved.setdefault('audio', (k['label'], k['sample_rate'], a[0])),
    )
    monkeypatch.setattr(
        tts_path, 'save_transcript', lambda *a, **k: saved.setdefault('text', (k['label'], a[0]))
    )


def test_run_scenario_は_B_と同じ_Run_を返して_google_の名前で残す(monkeypatch):
    saved: dict = {}
    _patch_pipeline(monkeypatch, _synth_handler([]), saved)

    run = google_path.run_scenario(
        SCENARIO, env={'GOOGLE_TTS_API_KEY': 'k', 'GEMINI_API_KEY': 'g'}
    )

    assert run.path == google_path.PATH_GOOGLE == 'google'
    assert run.model == VOICE
    assert run.credits == 0  # ElevenLabs のクレジットは使わない
    assert run.usd > 0
    assert run.correct is None
    assert run.first_audio_ms >= 0 and run.reply_done_ms >= run.first_audio_ms
    assert '検索1位=デプロイ手順' in run.note
    assert '$' in run.note
    label, sample_rate, pcm = saved['audio']
    assert (label, sample_rate) == ('google', 16000)
    assert pcm == b'\x01\x02\x03\x04'  # ヘッダは落ちている
    label, text = saved['text']
    assert label == 'google'
    assert f'構成: F（検索 → gemini-3.6-flash → Google {VOICE}）' in text
    assert 'ElevenLabs のクレジットは 0' in text
    assert '## Google の文ごとの応答' in text


def test_run_scenario_は枠切れなら記録せずに止まる(monkeypatch):
    saved: dict = {}
    _patch_pipeline(monkeypatch, _synth_handler([], status=429, error=QUOTA_ERROR), saved)

    with pytest.raises(google_path.GooglePathError, match='割り当て'):
        google_path.run_scenario(SCENARIO, env={'GOOGLE_TTS_API_KEY': 'k', 'GEMINI_API_KEY': 'g'})
    assert saved == {}


def test_run_scenario_は鍵が無ければ接続前に止まる():
    from voicelab.config import ConfigError

    with pytest.raises(ConfigError, match='GOOGLE_TTS_API_KEY'):
        google_path.run_scenario(SCENARIO, env={'GEMINI_API_KEY': 'g'})


def test_dry_run_は接続せず_B_と同じ検索結果を見せる():
    text = google_path.describe_dry_run(SCENARIO, env={})
    assert 'デプロイ手順' in text
    assert VOICE in text
    assert 'linear16_16000' in text
    assert '未設定' in text
    assert 'ストリーミングしない' in text


# --------------------------------------------------------------------------- 登録と表


def test_登録簿に_google_が入り表の見出しが出る():
    from voicelab import tts_provider

    provider = tts_provider.get('google')
    assert provider.label == 'B: 自前構成（Google）'
    assert provider.required_env == ('GOOGLE_TTS_API_KEY', 'GEMINI_API_KEY')
    assert 'google' in metrics.known_paths()
    run = metrics.Run(
        scenario_id='deploy-check', path='google',
        first_audio_ms=900, reply_done_ms=2000, credits=0,
    )
    assert '| B: 自前構成（Google） | 1 |' in metrics.render_report([run])
