"""A: Agents Platform で 1 往復して、遅延と費用を記録する。

**責務はここだけ**。「質問を 1 つ投げて、返事を最後まで受け取り、時刻とクレジットを残す」。
質問の選び方（1 問か 5 問か）、残高の見張り、表の更新は :mod:`voicelab.cli` の仕事。

計測の考え方:

- 質問は**テキストで送る**（``send_user_message``）。マイクを使うと、部屋の雑音と
  無音判定が毎回ちがう遅延を生んで、B と比べられなくなる。代わりに、この数字には
  **音声認識の時間が含まれない**。B も同じ条件（テキスト投入）で測ること。
- ``first_audio_ms`` は「送ってから最初の音の断片が届くまで」。体感の遅延はこれ。
- ``reply_done_ms`` は「送ってから最後の音の断片が届くまで」。
- SDK の ``callback_latency_measurement`` は使わない。あれは WebSocket の ping（ms）で、
  会話の遅延ではないため。

``audio_interface`` を省くとテキスト専用（音声なし）になるので、必ず
:class:`MeasuringAudioInterface` を渡す。鳴らさずに時刻と PCM を溜めるだけの実装で、
スピーカーもマイクも要らない。
"""

import json
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import AudioInterface, ClientTools, Conversation

from . import search
from .agent_setup import API_BASE, TOOL_NAME
from .config import AUDIO_DIR, COMPANY_ELEVENLABS, MODEL_AGENTS_PLATFORM, TRANSCRIPTS_DIR, load_env, require, result_dirs
from .metrics import PATH_AGENTS, Run

#: 最初の音が来てからこれだけ途切れたら「言い終わった」とみなす。
SILENCE_END_SECONDS = 1.5

#: 送ってからこれだけ音が来なければ諦める。記録は残す（note に書く）。
FIRST_AUDIO_TIMEOUT_SECONDS = 40.0

#: 接続（conversation_id の確定）を待つ上限。
CONNECT_TIMEOUT_SECONDS = 10.0

#: 会話の音声フォーマット。SDK の ``AudioInterface`` の契約どおり。
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

#: 無音を送るときの 1 回分（4000 サンプル = 250ms）。
SILENCE_CHUNK = b'\x00' * (4000 * SAMPLE_WIDTH)
SILENCE_INTERVAL_SECONDS = 0.25


class AgentsPathError(RuntimeError):
    """会話を始められなかった、または続けられなかった。"""


@dataclass
class ToolCall:
    """Agent が ``search_notes`` を呼んだ 1 回の記録。

    :ivar at: 呼ばれた時刻（``perf_counter``）。
    :ivar query: LLM が作った検索語。
    :ivar hits: 返した件数。
    :ivar top_note: 1 位のノート名（0 件なら None）。
    """

    at: float
    query: str
    hits: int
    top_note: str | None


def elapsed_ms(t0: float, at: float) -> int:
    """基準時刻からの経過をミリ秒（四捨五入）にする。"""
    return round((at - t0) * 1000)


def audio_window_ms(t0: float, times: list[float]) -> tuple[int, int]:
    """音の断片が届いた時刻の列から ``(最初の音, 最後の音)`` を ms で出す。

    1 つも届いていなければ ``(0, 0)``。**純粋関数にしてあるのはテストのため**で、
    実際の会話を流さずに終了判定と計算だけを確かめられるようにしている。
    """
    if not times:
        return 0, 0
    return elapsed_ms(t0, times[0]), elapsed_ms(t0, times[-1])


def reply_state(
    t0: float,
    now: float,
    times: list[float],
    *,
    silence_seconds: float = SILENCE_END_SECONDS,
    first_audio_timeout_seconds: float = FIRST_AUDIO_TIMEOUT_SECONDS,
) -> str:
    """今どの段階か。``waiting`` / ``speaking`` / ``done`` / ``timeout``。

    「言い終わり」をサーバの終了イベントではなく**音の途切れ**で決めているのは、
    B（自前構成）でも同じ判定が書けるようにするため。判定の仕方が違うと数字が比べられない。
    """
    if not times:
        return 'timeout' if now - t0 >= first_audio_timeout_seconds else 'waiting'
    return 'done' if now - times[-1] >= silence_seconds else 'speaking'


class MeasuringAudioInterface(AudioInterface):
    """鳴らさずに測る ``AudioInterface``。

    - ``output()`` … 届いた時刻を ``times`` に、PCM を ``chunks`` に積む。再生はしない
    - ``start()`` … ``input_callback`` を預かるだけ。マイクは開かない
    - ``interrupt()`` … 溜めた音は**捨てない**。捨てると録音が途中で欠けて、
      あとから聞いて正誤を判定できなくなる。印（``interrupted``）だけ立てる

    ``send_silence=True`` にすると、250ms ごとに無音を送るスレッドを立てる。
    テキストだけ送って Agent が黙る場合の逃げ道（README の「無音送出」を参照）。
    """

    def __init__(self, *, send_silence: bool = False, clock: Callable[[], float] = time.perf_counter):
        self.chunks: list[bytes] = []
        self.times: list[float] = []
        self.interrupted = False
        self._clock = clock
        self._send_silence = send_silence
        self._input_callback: Callable[[bytes], None] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        self._input_callback = input_callback
        if self._send_silence:
            self._thread = threading.Thread(target=self._feed_silence, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def output(self, audio: bytes) -> None:
        self.times.append(self._clock())
        self.chunks.append(audio)

    def interrupt(self) -> None:
        self.interrupted = True

    def _feed_silence(self) -> None:
        while not self._stop.is_set():
            callback = self._input_callback
            if callback is not None:
                try:
                    callback(SILENCE_CHUNK)
                except Exception:  # 接続が閉じた後の呼び出し。計測を止める理由にはしない
                    return
            self._stop.wait(SILENCE_INTERVAL_SECONDS)

    def pcm(self) -> bytes:
        """届いた断片をつないだ生 PCM。"""
        return b''.join(self.chunks)


@dataclass
class Capture:
    """1 往復のあいだに拾ったもの。

    :ivar responses: Agent の返答（本文）。
    :ivar transcripts: 送った発言の書き起こし（テキスト投入でも返ってくる）。
    :ivar tool_calls: ``search_notes`` の呼び出し。
    """

    responses: list[str] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)


def utc_stamp() -> str:
    """ファイル名に入れる UTC の時刻印。**C も同じものを使う**（並べ替えの基準を揃えるため）。"""
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def make_tool_handler(capture: Capture, clock: Callable[[], float] = time.perf_counter):
    """``search_notes`` の handler を作る。呼ばれた事実を ``capture`` に残す。

    戻り値は :func:`voicelab.search.search_notes` の JSON 文字列をそのまま返す。
    LLM に渡る内容と、こちらが記録する内容を**同じもの**にしておくため。
    """

    def handler(parameters: dict) -> str:
        query = str(parameters.get('query', ''))
        payload = search.search_notes(query)
        results = json.loads(payload)['results']
        capture.tool_calls.append(
            ToolCall(
                at=clock(),
                query=query,
                hits=len(results),
                top_note=results[0]['note'] if results else None,
            )
        )
        return payload

    return handler


def fetch_conversation_metadata(
    api_key: str, conversation_id: str, *, wait_seconds: float = 30.0
) -> dict | None:
    """会話が確定するまで待って ``metadata`` をそのまま返す（待ちきれなければ None）。

    終了直後は ``status`` が ``in-progress`` のまま費用が確定していないことがあるので、
    ``done`` になるまで 2 秒間隔で待つ。

    **辞書のまま返すのは C（Knowledge Base）のため**。C は費用と通話秒数に加えて
    ``rag_usage`` を書き起こしに残す（ElevenLabs 側の RAG が実際に引かれた証拠になる）。
    A が要るのは費用と秒数だけなので、そちらは :func:`fetch_conversation_cost` が包む。
    """
    deadline = time.monotonic() + wait_seconds
    headers = {'xi-api-key': api_key, 'accept': 'application/json'}
    with httpx.Client(headers=headers, timeout=30) as client:
        while True:
            try:
                response = client.get(f'{API_BASE}/convai/conversations/{conversation_id}')
            except httpx.HTTPError:
                return None
            if response.status_code < 400:
                payload = response.json()
                if payload.get('status') == 'done':
                    return payload.get('metadata') or {}
            if time.monotonic() >= deadline:
                return None
            time.sleep(2)


def conversation_cost(metadata: dict | None) -> tuple[int | None, int | None]:
    """``metadata`` から ``(credits, call_duration_secs)`` を取り出す純粋関数。"""
    if metadata is None:
        return None, None
    cost = metadata.get('cost')
    duration = metadata.get('call_duration_secs')
    return (
        int(cost) if cost is not None else None,
        int(duration) if duration is not None else None,
    )


def fetch_conversation_cost(
    api_key: str, conversation_id: str, *, wait_seconds: float = 30.0
) -> tuple[int | None, int | None]:
    """会話の費用と通話秒数を取る。``(credits, call_duration_secs)``。"""
    return conversation_cost(
        fetch_conversation_metadata(api_key, conversation_id, wait_seconds=wait_seconds)
    )


def save_audio_file(
    pcm: bytes,
    scenario_id: str,
    stamp: str,
    directory: Path = AUDIO_DIR,
    *,
    label: str = PATH_AGENTS,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """届いた PCM を WAV にする（既定は 16kHz / 16bit / mono）。

    正誤は**人が聞いて**判定する決まりなので、音は捨てずに残す。

    ``label`` は名前に入れる構成名（``agents`` / ``custom`` / ``deepgram`` …）。**B も同じ関数を使う**。
    形式や置き場所が構成ごとに違うと、並べて聞いたときに比べられなくなるため。
    ``sample_rate`` は 16kHz 以外しか返さない TTS（OpenAI は 24kHz）のためにある。
    変換せずそのまま書く。
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{scenario_id}_{label}_{stamp}.wav'
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return path


def render_transcript(
    scenario: dict,
    capture: Capture,
    *,
    first_audio_ms: int,
    reply_done_ms: int,
    conversation_id: str | None,
    credits: int | None,
    duration_secs: int | None,
    interrupted: bool,
    rag_usage: dict | None = None,
) -> str:
    """会話 1 回分を、音を聞かなくても追える形の文章にする。

    :param rag_usage: 会話メタデータの ``rag_usage``。**C（Knowledge Base）専用**で、
        渡すと「## RAG」の節が増える。A・B は渡さないので出力は変わらない。
    """
    lines = [
        f'質問: {scenario["text"]}（{scenario["id"]}）',
        f'期待するノート: {scenario.get("expected_note")}',
        f'会話 id: {conversation_id or "（取得できず）"}',
        '',
        '## ツール呼び出し',
    ]
    if capture.tool_calls:
        for call in capture.tool_calls:
            lines.append(f'- {TOOL_NAME}(query={call.query!r}) → {call.hits} 件 / 1位={call.top_note}')
    else:
        lines.append('- 呼ばれなかった')
    if rag_usage is not None:
        lines += [
            '',
            '## RAG（ElevenLabs 側の検索）',
            f'- rag_usage: {json.dumps(rag_usage, ensure_ascii=False)}',
        ]
    lines += [
        '',
        '## 書き起こし（こちらの発言）',
        *(f'- {t}' for t in capture.transcripts or ['（なし）']),
        '',
        '## 返答',
        *(capture.responses or ['（なし）']),
        '',
        '## 時刻',
        f'- 最初の音まで: {first_audio_ms} ms',
        f'- 言い終わりまで: {reply_done_ms} ms',
        f'- 通話秒数: {duration_secs if duration_secs is not None else "（取得できず）"}',
        f'- クレジット: {credits if credits is not None else "（取得できず）"}',
        f'- 割り込み: {"あり" if interrupted else "なし"}',
    ]
    return '\n'.join(lines) + '\n'


def save_transcript(
    text: str,
    scenario_id: str,
    stamp: str,
    directory: Path = TRANSCRIPTS_DIR,
    *,
    label: str = PATH_AGENTS,
) -> Path:
    """書き起こしを置く。``label`` の意味は :func:`save_audio_file` と同じ。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{scenario_id}_{label}_{stamp}.txt'
    path.write_text(text, encoding='utf-8')
    return path


def build_note(
    scenario: dict,
    capture: Capture,
    *,
    interrupted: bool,
    timed_out: bool,
    cost_missing: bool,
    lead: str | None = None,
) -> str:
    """CSV の 1 列に収まる長さで、あとから見て困らないだけのことを書く。

    :param lead: 先頭の一言を差し替える。**C（Knowledge Base）専用**で、道具を持たない C では
        「ツール呼ばなかった」と書いても何も伝わらないため、代わりに RAG の使われ方を入れる。
        既定の None なら今までどおりツールの呼び出し状況を書く。
    """
    parts: list[str] = []
    if lead is not None:
        parts.append(lead)
    elif capture.tool_calls:
        top = capture.tool_calls[0].top_note
        parts.append(f'ツール呼んだ({len(capture.tool_calls)}回) 1位={top}')
    else:
        parts.append('ツール呼ばなかった')
    parts.append(f'期待={scenario.get("expected_note")}')
    reply = ' '.join(capture.responses).replace('\n', ' ')
    parts.append(f'返答={reply[:40] or "（なし）"}')
    parts.append(f'割り込み={"あり" if interrupted else "なし"}')
    if timed_out:
        parts.append('音声が来ずタイムアウト')
    if cost_missing:
        parts.append('費用未取得')
    return ' / '.join(parts)


def describe_dry_run(scenario: dict, env: dict[str, str] | None = None) -> str:
    """接続せずに、何をするつもりかと検索結果だけを見せる。

    クレジットを 1 も使わずに「ツールが期待のノートを引けるか」を確かめられるようにするため。
    ここで期待外れなら、会話を流す価値がない。
    """
    env = load_env() if env is None else env
    hits = search.search(scenario['text'])
    lines = [
        f'[dry-run] {scenario["id"]}: {scenario["text"]}',
        f'  Agent: {env.get("ELEVENLABS_AGENT_ID") or "（未設定。setup-agent を先に実行）"}',
        f'  声/モデル: {env.get("ELEVENLABS_VOICE_ID", "")} / {env.get("ELEVENLABS_MODEL_ID", "")}',
        f'  期待するノート: {scenario.get("expected_note")}',
        f'  ローカル検索の結果（{TOOL_NAME} が返すもの）:',
    ]
    if hits:
        for index, hit in enumerate(hits, start=1):
            heading = hit['heading'] or '（見出しなし）'
            lines.append(f'    {index}. {hit["note"]} / {heading}')
    else:
        lines.append('    （0 件。Agent は「ノートには見当たりません」と答えるはず）')
    lines.append('  実行すると: 会話を 1 回開き、この質問をテキストで送り、音が 1.5 秒途切れたら終える')
    return '\n'.join(lines)


def run_scenario(
    scenario: dict,
    *,
    save_audio: bool = True,
    env: dict[str, str] | None = None,
    send_silence: bool = False,
) -> Run:
    """質問を 1 つ投げて 1 往復し、:class:`voicelab.metrics.Run` を返す。

    **クレジットを消費する。** 呼ぶ前に残高を確かめること（``voicelab run agents`` がやる）。

    :param save_audio: 返ってきた音声を WAV に残すか。
    :param send_silence: 250ms ごとに無音を送るか。テキストだけで応答が始まるなら不要。
    :raises AgentsPathError: 接続できなかった。
    """
    env = load_env() if env is None else env
    api_key = require('ELEVENLABS_API_KEY', env)
    agent_id = require('ELEVENLABS_AGENT_ID', env)

    capture = Capture()
    audio = MeasuringAudioInterface(send_silence=send_silence)
    tools = ClientTools()
    tools.register(TOOL_NAME, make_tool_handler(capture))

    conversation = Conversation(
        ElevenLabs(api_key=api_key),
        agent_id,
        requires_auth=True,
        audio_interface=audio,
        client_tools=tools,
        callback_agent_response=capture.responses.append,
        callback_user_transcript=capture.transcripts.append,
    )

    conversation.start_session()
    try:
        wait_for_connection(conversation)
        t0 = time.perf_counter()
        conversation.send_user_message(scenario['text'])
        state = wait_for_reply(t0, audio)
    finally:
        conversation.end_session()
        conversation_id = conversation.wait_for_session_end()

    first_audio_ms, reply_done_ms = audio_window_ms(t0, audio.times)
    credits_used, duration_secs = (
        fetch_conversation_cost(api_key, conversation_id) if conversation_id else (None, None)
    )

    stamp = utc_stamp()
    dirs = result_dirs(COMPANY_ELEVENLABS, MODEL_AGENTS_PLATFORM)
    if save_audio and audio.chunks:
        save_audio_file(audio.pcm(), scenario['id'], stamp, dirs.audio)
    save_transcript(
        render_transcript(
            scenario,
            capture,
            first_audio_ms=first_audio_ms,
            reply_done_ms=reply_done_ms,
            conversation_id=conversation_id,
            credits=credits_used,
            duration_secs=duration_secs,
            interrupted=audio.interrupted,
        ),
        scenario['id'],
        stamp,
        dirs.transcripts,
    )

    return Run(
        scenario_id=scenario['id'],
        path=PATH_AGENTS,
        model=MODEL_AGENTS_PLATFORM,
        first_audio_ms=first_audio_ms,
        reply_done_ms=reply_done_ms,
        credits=credits_used or 0,
        correct=None,  # 人が音を聞いて判定する
        note=build_note(
            scenario,
            capture,
            interrupted=audio.interrupted,
            timed_out=state == 'timeout',
            cost_missing=credits_used is None,
        ),
    )


def wait_for_connection(conversation: Conversation) -> None:
    """conversation_id が入るまで待つ。

    SDK には「つながったか」を知る公開の手段が無く、``start_session()`` は背景スレッドを
    立てて即座に返る。つながる前に ``send_user_message`` を呼ぶと
    ``RuntimeError: Session not started`` になるため、やむを得ず private の
    ``_conversation_id`` を覗いている（サーバの初期メタデータで埋まる）。

    :raises AgentsPathError: 制限時間内につながらなかった。
    """
    deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS
    while conversation._conversation_id is None:
        if time.monotonic() >= deadline:
            raise AgentsPathError(
                f'{CONNECT_TIMEOUT_SECONDS:.0f} 秒で接続できませんでした。'
                ' API キーと ELEVENLABS_AGENT_ID、Agent の enable_auth を確認してください。'
            )
        time.sleep(0.05)


def wait_for_reply(t0: float, audio: MeasuringAudioInterface) -> str:
    """返事が終わる（か、来ない）まで待つ。戻り値は ``done`` か ``timeout``。"""
    while True:
        state = reply_state(t0, time.perf_counter(), audio.times)
        if state in ('done', 'timeout'):
            return state
        time.sleep(0.05)


if __name__ == '__main__':
    # python -m voicelab.agents_path rollback
    # **クレジットを消費する。** 残高の見張りは cli 側にあるので、ここには無い。
    import sys

    from .config import find_scenario

    print(run_scenario(find_scenario(sys.argv[1] if len(sys.argv) > 1 else 'rollback')))
