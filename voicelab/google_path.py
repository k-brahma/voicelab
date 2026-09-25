"""B の TTS を **Google Cloud Text-to-Speech（Chirp 3 HD）** に差し替えた構成。

検索・Gemini の呼び方・``t0`` の打ち方・文の切り出し・時刻の記録は B の
:func:`~voicelab.custom_path.run_pipeline` と :func:`~voicelab.custom_path.run_turn` を
**そのまま使う**。1 往復の流し方と記録は :mod:`voicelab.tts_path` にあり、ここに置くのは
Google に固有の 3 つ（:class:`GoogleTts` の送受信、料金、残高の注記）だけ。

B（ElevenLabs）・Deepgram との違い（これが比べたいもの）:

- TTS が ``POST /v1/text:synthesize``（REST、**1 文 1 リクエスト**）になる。ここまでは
  Deepgram と同じだが、Google の REST は**ストリーミングしない**。応答は JSON で、音は
  ``audioContent`` に base64 で**1 文ぶんまとめて**入って届く。ストリーミングの
  ``streamingSynthesize`` は gRPC にしか無く、ここでは使わない（依存を増やさず ``httpx`` だけで呼ぶ）
- そのため「最初のバイト」は**応答が丸ごと届いた時刻**になる。1 文目の合成が終わるまで
  音は 1 バイトも来ないので、``first_audio_ms`` には 1 文目の合成時間がまるごと乗る。
  **この作りの違い（ストリーミングの有無）も比べる対象に含まれる**。Google を遅いと読む前に、
  それが声の合成の速さなのか、届け方の違いなのかを分けて考えること
- 出力は ``LINEAR16`` / 16kHz。B の ``pcm_16000`` と同じ形だが、応答の先頭に **44 バイトの
  WAV ヘッダ**（``RIFF`` … ``data``）が付いてくる。文ごとにつなぐとヘッダがプツッと鳴るので、
  :func:`strip_wav_header` で落としてから :class:`~voicelab.custom_path.AudioSink` に入れる
- 費用は ElevenLabs のクレジットではなく**ドル**（Chirp 3 HD は 1,000 文字 $0.030。
  **月 100 万文字までは無料**）。``credits`` 列は 0、ドルは ``usd`` 列と ``note`` 列と書き起こしに書く。
  無料枠の内側でも ``usd`` 列には枠が無いときの額を書く（他社と並べるため）

2026-09-25 に ``POST /v1/text:synthesize`` を 1 回だけ投げて（「テストです。」、Kore）確かめた形:
HTTP 200、``Content-Type: application/json``、キーは ``audioContent`` だけ。base64 35,916 文字 →
26,936 バイト。先頭 44 バイトが ``RIFF``/``WAVE``/``fmt``（mono・16000Hz・16bit）/``data``
（26,892 バイト ≒ 0.84 秒）。応答まで 608ms（接続込み）。

日本語の Chirp 3 HD の声は 2026-09-25 に ``GET /v1/voices?languageCode=ja-JP`` で確かめて 30 声
（ja-JP 全体は 41 声。どれも ``naturalSampleRateHertz`` は 24000 で、16000 を頼めば 16000 で返る）。
名前はどれも ``ja-JP-Chirp3-HD-<名>``:

- 女性（14）: Achernar / Aoede / Autonoe / Callirrhoe / Despina / Erinome / Gacrux /
  **Kore（既定）** / Laomedeia / Leda / Pulcherrima / Sulafat / Vindemiatrix / Zephyr
- 男性（16）: Achird / Algenib / Algieba / Alnilam / Charon / Enceladus / Fenrir / Iapetus /
  Orus / Puck / Rasalgethi / Sadachbia / Sadaltager / Schedar / Umbriel / Zubenelgenubi

既定を Kore にしたのは、B（ElevenLabs）・Deepgram（izanami）の既定と同じ女性の声で、
Google の Chirp 3 HD の説明で例に出てくることが多い声だから。替えるなら ``.env`` の
``GOOGLE_TTS_VOICE`` に上の名前を書く。
"""

import base64
import queue
import struct
import threading
import time
from typing import Callable

import httpx

from . import tts_path, tts_provider
from .custom_path import AudioSink, CustomPathError, Turn

#: CSV の ``path`` 列と CLI の ``--tts`` の値。**一度決めた綴りは変えない。**
PATH_GOOGLE = 'google'

#: ``results/google/<声>/`` の会社の段。
COMPANY = 'google'

#: Google Cloud Text-to-Speech の REST の根。合成は ``/v1/text:synthesize``、声の一覧は ``/v1/voices``。
API_BASE = 'https://texttospeech.googleapis.com'

#: 声の言語。Chirp 3 HD の声は言語ごとに名前が分かれている（``ja-JP-Chirp3-HD-…``）。
LANGUAGE_CODE = 'ja-JP'

#: 既定の声（女性。一覧はモジュールの説明にある）。
DEFAULT_GOOGLE_TTS_VOICE = 'ja-JP-Chirp3-HD-Kore'

#: 出力の形。B の ``pcm_16000``（16kHz / 16bit / mono）に揃える。
#: ``LINEAR16`` は WAV のヘッダ付きで返るので、:func:`strip_wav_header` で落とす。
ENCODING = 'LINEAR16'
SAMPLE_RATE = 16000

#: 書き起こしと dry-run に出す出力形式の名前。
OUTPUT_FORMAT = f'{ENCODING.lower()}_{SAMPLE_RATE}'

#: Chirp 3 HD の料金（1,000 文字あたりのドル）。
#:
#: 出典: https://cloud.google.com/text-to-speech/pricing （2026-09-25 確認。Chirp 3: HD voices は
#: 月 0〜100 万文字が無料、超えたら 1 文字 $0.00003 = 100 万文字 $30。Instant custom voice は $60）。
#: これは**見積り**で、無料枠の内側なら実際の請求は 0。正は Cloud Console の課金の画面。
USD_PER_1K_CHARS = 0.030

#: 1 リクエストの待ち上限（秒）。接続・読み取りとも。
REQUEST_TIMEOUT_SECONDS = 30.0


class GooglePathError(CustomPathError):
    """1 往復を始められなかった、または続けられなかった。

    :class:`voicelab.custom_path.CustomPathError` を継いでいるのは、CLI の受け口を
    増やさずに B と同じ扱いにするため。
    """


# --------------------------------------------------------------------------- 費用


def estimate_usd(sent_chars: int, usd_per_1k: float = USD_PER_1K_CHARS) -> float:
    """TTS に送った文字数からドルを見積もる。Chirp 3 HD は**文字数**の課金（無料枠は数えない）。"""
    return tts_path.usd_per_chars(sent_chars, usd_per_1k)


# --------------------------------------------------------------------------- TTS


def synthesize_url(*, base: str = API_BASE) -> str:
    """``POST /v1/text:synthesize`` の URL。声と出力の形は本文で渡す。"""
    return f'{base}/v1/text:synthesize'


def voices_url(*, base: str = API_BASE) -> str:
    """``GET /v1/voices`` の URL（日本語だけ）。接続を温めるのに使う。課金なし。"""
    return f'{base}/v1/voices?languageCode={LANGUAGE_CODE}'


def synthesize_body(sentence: str, voice: str) -> dict:
    """``/v1/text:synthesize`` に送る本文。B と違って末尾の空白は要らない（ElevenLabs 固有の決まり）。"""
    return {
        'input': {'text': sentence},
        'voice': {'languageCode': LANGUAGE_CODE, 'name': voice},
        'audioConfig': {'audioEncoding': ENCODING, 'sampleRateHertz': SAMPLE_RATE},
    }


def strip_wav_header(audio: bytes) -> bytes:
    """``LINEAR16`` の応答から WAV のヘッダを落として、生 PCM だけにする。

    先頭が ``RIFF`` … ``WAVE`` なら、チャンクをたどって ``data`` の中身を返す（ふつうは
    44 バイト目から）。``RIFF`` で始まらなければ、もう生 PCM なのでそのまま返す。
    ``data`` が見つからないときも、壊すよりはましなので、そのまま返す。
    """
    if len(audio) < 12 or audio[:4] != b'RIFF' or audio[8:12] != b'WAVE':
        return audio
    offset = 12
    while offset + 8 <= len(audio):
        tag = audio[offset:offset + 4]
        (size,) = struct.unpack('<I', audio[offset + 4:offset + 8])
        body = offset + 8
        if tag == b'data':
            return audio[body:body + size]
        offset = body + size + (size & 1)  # チャンクは偶数バイトに揃えてある
    return audio


def decode_audio(payload: dict) -> bytes:
    """応答の JSON から音を取り出す。base64 を戻し、WAV のヘッダを落とす。"""
    return strip_wav_header(base64.b64decode(payload['audioContent']))


def describe_http_error(response: httpx.Response) -> str:
    """失敗した応答を 1 行に。``HTTP 403 PERMISSION_DENIED: <message>`` の形にする。

    Google は ``{"error": {"code", "message", "status"}}`` を返す。本文をそのまま切ると
    肝心の ``message`` が途中で切れることがあるので、中身を取り出してから縮める。
    """
    status = ''
    message = response.text
    try:
        error = response.json().get('error', {})
        status = error.get('status') or ''
        message = error.get('message') or message
    except (ValueError, AttributeError):
        pass
    head = f'HTTP {response.status_code}' + (f' {status}' if status else '')
    return f'{head}: {message[:200]}'


def _make_client(api_key: str) -> httpx.Client:
    return httpx.Client(
        headers={'x-goog-api-key': api_key},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


class GoogleTts:
    """文が確定するたびに ``POST /v1/text:synthesize`` を投げ、1 文ぶんの音をまとめて受ける。

    :class:`voicelab.custom_path.TtsStream` を満たすので、B の ``run_turn`` にそのまま渡せる。

    **1 文 1 リクエスト**にしているのは Deepgram と同じ理由（REST には「1 本の接続に文を
    流し込む」口が無い）。LLM の生成を止めないよう、``send()`` は文を列に積むだけですぐ戻り、
    送信と受信は別スレッドが**文の順に 1 本ずつ**行う（音の順番を崩さないため）。

    Deepgram と違って応答は**ストリーミングされない**。JSON が丸ごと届いてから base64 を戻し、
    ヘッダを落とした PCM を :class:`AudioSink` に **1 回の** ``add()`` で入れる。だから文ごとの
    ``ttfb_ms`` は「リクエストから応答が全部届くまで」で、その文の合成時間をまるごと含む。

    ``open()`` で 1 度 ``GET /v1/voices``（課金なし）を投げて、TCP と TLS の接続を
    ``t0`` より**前**に張っておく。B が WebSocket の接続を ``t0`` 前に済ませているのと
    揃えるため。これをしないと、最初の文の遅延に接続の時間が乗る。
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice: str,
        sink: AudioSink,
        client_factory: Callable[[str], httpx.Client] = _make_client,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._api_key = api_key
        self._voice = voice
        self._sink = sink
        self._client_factory = client_factory
        self._clock = clock
        self._client: httpx.Client | None = None
        self._queue: 'queue.Queue[str | None]' = queue.Queue()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self.sent_sentences = 0
        self.sent_chars = 0
        self.first_send_at: float | None = None
        #: 文ごとの「リクエストを投げてから応答が全部届くまで」（ms）。書き起こしに書く。
        self.ttfb_ms: list[int] = []
        self.error: str | None = None

    @property
    def url(self) -> str:
        return synthesize_url()

    def open(self) -> 'GoogleTts':
        """接続を温めて、送受信のスレッドを立てる。**t0 より前に呼ぶこと。**"""
        try:
            self._client = self._client_factory(self._api_key)
            response = self._client.get(voices_url())
        except httpx.HTTPError as exc:
            raise GooglePathError(f'Google に接続できませんでした: {type(exc).__name__}') from exc
        if response.status_code >= 400:
            raise GooglePathError(
                f'Google が {describe_http_error(response)} を返しました'
                '（鍵と Cloud Text-to-Speech API の有効化を確かめる）'
            )
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()
        return self

    def _work(self) -> None:
        """列から文を取り、1 本ずつ投げて音を受ける。届いた瞬間に :class:`AudioSink` が時刻を打つ。"""
        try:
            while True:
                sentence = self._queue.get()
                if sentence is None:
                    break
                started = self._clock()
                response = self._client.post(self.url, json=synthesize_body(sentence, self._voice))
                if response.status_code >= 400:
                    self.error = describe_http_error(response)
                    break
                pcm = decode_audio(response.json())
                if not pcm:
                    continue
                self._sink.add(pcm)
                self.ttfb_ms.append(round((self._sink.times[-1] - started) * 1000))
        except (KeyError, ValueError) as exc:  # JSON や base64 が壊れていたとき（どちらも ValueError の子）
            self.error = f'応答の形が想定と違います: {type(exc).__name__}: {exc}'
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self._done.set()

    def send(self, sentence: str) -> None:
        """文が 1 つ確定したら列に積む。**LLM の完了は待たない。**"""
        if self.first_send_at is None:
            self.first_send_at = self._clock()
        self.sent_sentences += 1
        self.sent_chars += len(sentence)
        self._queue.put(sentence)

    def finish(self) -> None:
        """「もう送らない」を伝える。列に残った文はこの後も順に処理される。"""
        self._queue.put(None)

    def wait(self, timeout: float) -> bool:
        """最後の文の音を受け終わる（か失敗する）まで待つ。時間切れなら False。"""
        return self._done.wait(timeout)

    def close(self) -> None:
        if self._thread is not None:
            self._queue.put(None)  # finish() を経ずに抜けたときの止め
            self._thread.join(timeout=2)
            self._thread = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> 'GoogleTts':
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()


def is_payment_required(error: str | None) -> bool:
    """TTS のエラーが「残高切れ／枠切れ」か。

    Google には前払いの残高が無いので、止まるのは課金が有効でない（HTTP 403 で本文に
    ``billing``）か、割り当て（quota）を使い切った（HTTP 403 で ``quota``、または
    HTTP 429 ``RESOURCE_EXHAUSTED``）とき。どちらも流し直しても音は出ないので、記録しない。
    ただの 403（API が無効、鍵の制限）は鍵の問題なので、ここでは拾わない。
    """
    if not error:
        return False
    if error.startswith('HTTP 429'):
        return True
    lowered = error.lower()
    return error.startswith('HTTP 403') and ('billing' in lowered or 'quota' in lowered)


# --------------------------------------------------------------------------- 1 往復（共通ランナーに渡すもの）


def _make_tts(api_key: str, voice: str, sink: AudioSink) -> GoogleTts:
    # モジュールの属性を毎回引くのは、テストが ``GoogleTts`` を差し替えられるようにするため
    return GoogleTts(api_key=api_key, voice=voice, sink=sink)


def _pricing_line(turn: Turn) -> str:
    return (
        f'- Google（見積り）: ${estimate_usd(turn.sent_chars):.5f}'
        f'（{turn.sent_chars} 字 × ${USD_PER_1K_CHARS} / 1,000 字。月 100 万字までの無料枠は数えない）。'
        'ElevenLabs のクレジットは 0'
    )


def _extra_transcript(tts: object) -> list[str]:
    ttfb = getattr(tts, 'ttfb_ms', None) or []
    lines = ['## Google の文ごとの応答（リクエストから応答が全部届くまで。ストリーミングしない）']
    lines += [f'- {index}. {ms} ms' for index, ms in enumerate(ttfb, start=1)] or ['- （なし）']
    return lines


SPEC = tts_path.TtsSpec(
    path=PATH_GOOGLE,
    company=COMPANY,
    name='Google',
    letter='F',
    api_key_env='GOOGLE_TTS_API_KEY',
    model_env='GOOGLE_TTS_VOICE',
    default_model=DEFAULT_GOOGLE_TTS_VOICE,
    output_format=OUTPUT_FORMAT,
    sample_rate=SAMPLE_RATE,
    make_tts=_make_tts,
    estimate_usd=lambda turn: estimate_usd(turn.sent_chars),
    pricing_line=_pricing_line,
    is_payment_required=is_payment_required,
    payment_message=(
        '課金か割り当てが尽きています（HTTP 403 billing/quota か 429）。'
        'Cloud Console で課金と割り当てを確かめてから流し直す。results/ には何も書いていません'
    ),
    extra_transcript=_extra_transcript,
    dry_run_action=(
        '検索 → LLM をストリーミング → 文が確定するたびに /v1/text:synthesize へ 1 本ずつ投げ、'
        '1 文ぶんの音をまとめて受ける（REST はストリーミングしない）'
    ),
    error_type=GooglePathError,
)


def build_google_note(scenario: dict, turn: Turn) -> str:
    """B の ``note`` にドルを足したもの。``credits`` 列が 0 なので、費用はここと ``usd`` 列で読む。"""
    return tts_path.build_usd_note(SPEC, scenario, turn)


class _TtfbHolder:
    """``render_transcript`` に文ごとの応答時間だけ渡すための入れ物。"""

    def __init__(self, ttfb_ms: list[int] | None):
        self.ttfb_ms = ttfb_ms or []


def render_transcript(
    scenario: dict, turn: Turn, *, model: str, llm: str, ttfb_ms: list[int] | None = None
) -> str:
    """B と同じ並びの書き起こしに、Google の文ごとの応答時間を足す。"""
    return tts_path.render_transcript(
        SPEC, scenario, turn, model=model, llm=llm, tts=_TtfbHolder(ttfb_ms)
    )


def describe_dry_run(scenario: dict, env: dict[str, str] | None = None) -> str:
    """接続せずに、何をするつもりかと検索結果を見せる（課金なし）。"""
    return tts_path.describe_dry_run(SPEC, scenario, env)


def run_scenario(scenario: dict, *, save_audio: bool = True, env: dict[str, str] | None = None):
    """質問を 1 つ投げて 1 往復し、:class:`voicelab.metrics.Run` を返す。

    **Google の文字数（無料枠を超えたらドル）と Gemini のトークンを消費する。**

    :raises GooglePathError: Google につなげなかった、または課金・割り当て切れ（記録しない）。
    :raises voicelab.config.ConfigError: 鍵が足りない。
    """
    return tts_path.run_scenario(SPEC, scenario, save_audio=save_audio, env=env)


def describe_balance(env: dict[str, str]) -> str:
    """残高の代わりの 1 行。API キーでは Google の課金を読めないので、見る場所だけ書く。"""
    return 'Google: 残高は API から読めない（Cloud Console の課金で見る。Chirp 3 HD は月 100 万文字まで無料）'


PROVIDER = tts_provider.register(
    tts_provider.TtsProvider(
        key=PATH_GOOGLE,
        name='Google',
        label='B: 自前構成（Google）',
        required_env=('GOOGLE_TTS_API_KEY', 'GEMINI_API_KEY'),
        run_scenario=run_scenario,
        describe_dry_run=describe_dry_run,
        describe_balance=describe_balance,
        cost_hint='ElevenLabs クレジット 0（ドルは usd 列。月 100 万文字までは実際には無料）',
        after_run_hints=(
            '文ごとの Google の応答時間と Gemini のトークンは results/google/<声>/transcripts/ にある。',
            'Google の REST はストリーミングしないので、first_audio_ms には 1 文目の合成時間がまるごと乗る。',
        ),
    )
)


__all__ = [
    'COMPANY',
    'DEFAULT_GOOGLE_TTS_VOICE',
    'GooglePathError',
    'GoogleTts',
    'PATH_GOOGLE',
    'PROVIDER',
    'SPEC',
    'USD_PER_1K_CHARS',
    'build_google_note',
    'decode_audio',
    'describe_balance',
    'describe_dry_run',
    'describe_http_error',
    'estimate_usd',
    'is_payment_required',
    'render_transcript',
    'run_scenario',
    'strip_wav_header',
    'synthesize_body',
    'synthesize_url',
    'voices_url',
]


if __name__ == '__main__':
    # python -m voicelab.google_path rollback
    # **Google の文字数と Gemini のトークンを消費する。**
    import sys

    from .config import find_scenario

    print(run_scenario(find_scenario(sys.argv[1] if len(sys.argv) > 1 else 'rollback')))
