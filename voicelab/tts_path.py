"""B（自前構成）の TTS を **REST 系のプロバイダ**に差し替えるときの共通部分。

Deepgram・OpenAI・Google のように「文を 1 つ POST して音を受ける」型の TTS は、
違うのが **リクエストの組み立て・出力の形・料金**だけで、1 往復の流し方（検索 → Gemini →
文が確定するたびに TTS）・時刻の打ち方・書き起こしの節の並びは B と同じにしなければ
数字を並べられない。ここに 1 往復の流し方を 1 つだけ置き、各プロバイダのモジュールは
:class:`TtsSpec` に「自分に固有のもの」を詰めて :func:`run_scenario` を呼ぶ。

プロバイダを 1 つ足す手順（Deepgram の :mod:`voicelab.deepgram_path` が見本）:

1. :class:`voicelab.custom_path.TtsStream` を満たすクラスを書く（``open`` で接続を温め、
   ``send`` で文を積み、``finish`` / ``wait`` / ``close``。音は :class:`~voicelab.custom_path.AudioSink`
   に入れ、``error`` に失敗を残す）
2. :class:`TtsSpec` を 1 つ作る（鍵のキー名、モデルのキー名と既定、出力の形、料金の換算）
3. :func:`voicelab.tts_provider.register` に :class:`~voicelab.tts_provider.TtsProvider` を渡す
4. :mod:`voicelab.cli` の ``PROVIDER_MODULES`` にモジュールを足す（import されないと登録されない）

計測の定義は A・B と同じ（README の「計測の定義」）。``t0`` は質問テキストを渡した時刻、
``first_audio_ms`` は最初の音声チャンクが届くまで、``reply_done_ms`` は最後のチャンクまで。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import search
from .agents_path import save_audio_file, save_transcript
from .config import DEFAULT_LLM, load_env, require, result_dirs
from .custom_path import (
    AudioSink,
    CustomPathError,
    Turn,
    build_note,
    build_system_prompt,
    render_turn_transcript,
    run_pipeline,
)
from .metrics import Run


def usd_per_chars(sent_chars: int, usd_per_1k: float) -> float:
    """文字数課金のプロバイダのドル。``sent_chars × 単価 / 1,000``。"""
    return sent_chars * usd_per_1k / 1000


@dataclass(frozen=True)
class TtsSpec:
    """REST 系プロバイダ 1 つぶんの「固有のもの」。

    :ivar path: CSV の ``path`` 列に入る値（= :attr:`voicelab.tts_provider.TtsProvider.key`）。
    :ivar company: ``results/<会社>/<モデル>/`` の会社の段（``deepgram`` / ``openai`` / ``google``）。
    :ivar name: 表示名（``Deepgram``）。
    :ivar letter: 書き起こしの「構成」行に出す記号（``D``）。
    :ivar api_key_env: 鍵の ``.env`` キー名。
    :ivar model_env: モデル（声）の ``.env`` キー名。
    :ivar default_model: ``.env`` に無いときのモデル。
    :ivar output_format: 書き起こしと dry-run に出す出力形式の名前（``linear16_16000``）。
    :ivar sample_rate: 受け取る PCM のサンプルレート。WAV に書くときに使う。
        16kHz 以外を返すプロバイダ（OpenAI は 24kHz）もそのまま残す。**変換はしない**
        （変換の時間が「言い終わり」に乗ると比べられなくなる）。
    :ivar make_tts: ``(api_key, model, sink)`` から TTS を作る。``open`` は :func:`run_pipeline` が呼ぶ。
    :ivar estimate_usd: 1 往復のドル。``(turn)`` で呼ぶ。文字数課金なら :func:`usd_per_chars`。
    :ivar pricing_line: 書き起こしの「費用」の節に出す 1 行。``(turn)`` で呼ぶ。
    :ivar is_payment_required: TTS の ``error`` が「残高切れ」か。True なら**記録せずに止める**
        （音 0 の行を CSV に積むと中央値が壊れる）。
    :ivar payment_message: 残高切れのときに出す文。
    :ivar extra_transcript: 書き起こしの末尾に足す節。``(tts)`` で呼び、行の一覧を返す。
    :ivar dry_run_action: dry-run の「実行すると」の文。
    :ivar error_type: 接続できなかったときに投げる例外（:class:`CustomPathError` の子）。
    :ivar folder_name: ``results/<会社>/<モデル>/`` のモデルの段。``(env, model)`` で呼ぶ。既定はモデル名。
    """

    path: str
    company: str
    name: str
    letter: str
    api_key_env: str
    model_env: str
    default_model: str
    output_format: str
    sample_rate: int
    make_tts: Callable[[str, str, AudioSink], object]
    estimate_usd: Callable[[Turn], float]
    pricing_line: Callable[[Turn], str]
    is_payment_required: Callable[[str | None], bool] = lambda _error: False
    payment_message: str = '残高がありません。クレジットを足してから流し直す。results/ には何も書いていません'
    extra_transcript: Callable[[object], list[str]] = field(default=lambda _tts: [])
    dry_run_action: str = '検索 → LLM をストリーミング → 文が確定するたびに TTS へ送り、音を受ける'
    error_type: type[CustomPathError] = CustomPathError
    #: ``results/<会社>/<ここ>/`` のモデルの段と ``Run.model``。既定はモデル名そのまま。
    #: モデルと声が別の設定になっている社（OpenAI）は ``<モデル>_<声>`` にして、声ごとに分ける。
    folder_name: Callable[[dict[str, str], str], str] = field(default=lambda _env, model: model)

    def model(self, env: dict[str, str]) -> str:
        return env.get(self.model_env) or self.default_model


def build_usd_note(spec: TtsSpec, scenario: dict, turn: Turn) -> str:
    """B の ``note`` にドルを足したもの。``credits`` 列が 0 なので、費用はここと ``usd`` 列で読む。"""
    return f'{build_note(scenario, turn)} / ${spec.estimate_usd(turn):.5f}'


def render_transcript(spec: TtsSpec, scenario: dict, turn: Turn, *, model: str, llm: str, tts=None) -> str:
    """B と同じ並びの書き起こしに、プロバイダ固有の節を足す。"""
    text = render_turn_transcript(
        scenario,
        turn,
        config_line=f'{spec.letter}（検索 → {llm} → {spec.name} {model}）',
        tts_cost_line=spec.pricing_line(turn),
        audio_format=spec.output_format,
    )
    extra = spec.extra_transcript(tts) if tts is not None else []
    if not extra:
        return text
    return text + '\n'.join(['', *extra]) + '\n'


def describe_dry_run(spec: TtsSpec, scenario: dict, env: dict[str, str] | None = None) -> str:
    """接続せずに、何をするつもりかと検索結果を見せる（課金なし）。"""
    env = load_env() if env is None else env
    hits = search.search(scenario['text'])
    prompt = build_system_prompt(hits)
    lines = [
        f'[dry-run] {scenario["id"]}: {scenario["text"]}',
        f'  LLM: {env.get("VOICELAB_LLM") or DEFAULT_LLM}'
        f'（鍵 GEMINI_API_KEY: {"あり" if env.get("GEMINI_API_KEY") else "未設定"}）',
        f'  TTS: {spec.name} {spec.model(env)} / {spec.output_format}'
        f'（鍵 {spec.api_key_env}: {"あり" if env.get(spec.api_key_env) else "未設定"}）',
        f'  期待するノート: {scenario.get("expected_note")}',
        '  検索の結果（B と同じ。LLM に選ばせず、先にこれを渡す）:',
    ]
    if hits:
        for index, hit in enumerate(hits, start=1):
            lines.append(f'    {index}. {hit["note"]} / {hit["heading"] or "（見出しなし）"}')
    else:
        lines.append('    （0 件。LLM に「ノートには見当たりません」と言わせる）')
    lines += [
        f'  system prompt: {len(prompt)} 文字（B と同じもの）',
        f'  実行すると: {spec.dry_run_action}',
    ]
    return '\n'.join(lines)


def run_scenario(
    spec: TtsSpec, scenario: dict, *, save_audio: bool = True, env: dict[str, str] | None = None
) -> Run:
    """質問を 1 つ投げて 1 往復し、:class:`voicelab.metrics.Run` を返す。

    **プロバイダの残高と Gemini のトークンを消費する。**

    :raises CustomPathError: （の子）TTS につなげなかった、または残高切れ。
    :raises voicelab.config.ConfigError: 鍵が足りない。
    """
    env = load_env() if env is None else env
    api_key = require(spec.api_key_env, env)
    gemini_key = require('GEMINI_API_KEY', env)
    model = spec.model(env)
    llm = env.get('VOICELAB_LLM') or DEFAULT_LLM

    sink = AudioSink()
    tts = spec.make_tts(api_key, model, sink)
    turn = run_pipeline(scenario['text'], tts=tts, sink=sink, gemini_key=gemini_key, llm=llm)
    if spec.is_payment_required(turn.error):
        raise spec.error_type(f'{spec.name} の{spec.payment_message}')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dirs = result_dirs(spec.company, spec.folder_name(env, model))
    if save_audio and turn.pcm:
        save_audio_file(
            turn.pcm, scenario['id'], stamp, dirs.audio, label=spec.path, sample_rate=spec.sample_rate
        )
    save_transcript(
        render_transcript(spec, scenario, turn, model=model, llm=llm, tts=tts),
        scenario['id'],
        stamp,
        dirs.transcripts,
        label=spec.path,
    )

    return Run(
        scenario_id=scenario['id'],
        path=spec.path,
        first_audio_ms=turn.first_audio_ms,
        reply_done_ms=turn.reply_done_ms,
        credits=0,  # ElevenLabs を使わない
        usd=round(spec.estimate_usd(turn), 6),
        model=dirs.model,
        correct=None,  # 人が音を聞いて判定する
        note=build_usd_note(spec, scenario, turn),
    )


__all__ = [
    'TtsSpec',
    'build_usd_note',
    'describe_dry_run',
    'render_transcript',
    'run_scenario',
    'usd_per_chars',
]
