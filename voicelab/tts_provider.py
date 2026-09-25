"""B（自前構成）で差し替えられる **TTS プロバイダの登録簿**。

自前構成（検索 → Gemini → TTS）は、TTS の部分だけを取り替えられる。ElevenLabs、
Deepgram、OpenAI、Google … と増やすとき、CLI・表・残高の見張りを毎回書き足さずに
済むよう、プロバイダごとに「名前・鍵・流し方・残高の読み方」をここに登録する。

登録するのは各プロバイダのモジュール（:mod:`voicelab.custom_path` = ElevenLabs、
:mod:`voicelab.deepgram_path` = Deepgram …）で、読み込まれたときに :func:`register` を呼ぶ。
CLI（:mod:`voicelab.cli`）はそれらを import してから :func:`all_providers` を見る。

このモジュールは**他の voicelab のモジュールを import しない**。登録簿が構成のモジュールを
知っていると循環参照になるため。
"""

from dataclasses import dataclass
from typing import Callable, Protocol


class RunLike(Protocol):
    """:class:`voicelab.metrics.Run` のこと。ここで import すると循環するので形だけ書く。"""

    scenario_id: str
    path: str
    first_audio_ms: int
    reply_done_ms: int
    credits: int
    usd: float
    note: str


@dataclass(frozen=True)
class TtsProvider:
    """B の TTS を 1 つ差し替えるのに要るもの一式。

    :ivar key: CSV の ``path`` 列に入る値、CLI の ``--tts`` の値。**一度決めた綴りは変えない**
        （過去の行が読めなくなる）。ElevenLabs は歴史的に ``custom``、Deepgram は ``deepgram``。
    :ivar name: 表示名（``ElevenLabs`` / ``Deepgram`` …）。
    :ivar label: 表の見出し（``B: 自前構成（Deepgram）``）。
    :ivar required_env: 実行前に揃っているべき ``.env`` のキー。TTS の鍵と Gemini の鍵。
    :ivar run_scenario: 質問 1 つを 1 往復流して ``Run`` を返す。**課金される。**
        ``(scenario, save_audio=..., env=...)`` で呼ぶ。
    :ivar describe_dry_run: 接続せずに、何をするつもりかを文章で返す。``(scenario, env=...)``。
    :ivar describe_balance: 残高を 1 行で返す（読めなければその旨）。``None`` なら表示しない。
        ``(env)`` で呼ぶ。
    :ivar uses_elevenlabs_credits: ElevenLabs のクレジットを消費するか。True なら CLI が
        :data:`voicelab.cli.MIN_CREDITS` の門をくぐらせる（A・B・C と同じ止まり方にするため）。
    :ivar cost_hint: CLI が 1 往復ごとに出す費用の言い方（``ElevenLabs クレジット（見積り）`` など）。
    :ivar after_run_hints: 実行後に出す案内（書き起こしに何があるか）。
    """

    key: str
    name: str
    label: str
    required_env: tuple[str, ...]
    run_scenario: Callable[..., RunLike]
    describe_dry_run: Callable[..., str]
    describe_balance: Callable[[dict[str, str]], str] | None = None
    uses_elevenlabs_credits: bool = False
    cost_hint: str = 'ドルは usd 列'
    after_run_hints: tuple[str, ...] = ()


_PROVIDERS: dict[str, TtsProvider] = {}


def register(provider: TtsProvider) -> TtsProvider:
    """登録する。同じ ``key`` を 2 度登録したら、それは書き間違いなので止める。

    ただし**同じモジュールの再 import**（テストの reload など）で同一の定義が来たときは通す。
    """
    existing = _PROVIDERS.get(provider.key)
    if existing is not None and existing is not provider and existing != provider:
        raise ValueError(f'TTS プロバイダ {provider.key!r} は登録済みです（{existing.name}）')
    _PROVIDERS[provider.key] = provider
    return provider


def get(key: str) -> TtsProvider:
    """``key`` のプロバイダ。無ければ、あるものを並べて止まる。"""
    try:
        return _PROVIDERS[key]
    except KeyError:
        known = ', '.join(sorted(_PROVIDERS)) or '（まだ何も登録されていない）'
        raise KeyError(f'TTS プロバイダ {key!r} はありません。あるのは: {known}') from None


def all_providers() -> tuple[TtsProvider, ...]:
    """登録順のプロバイダ一覧（CLI の選択肢と表の並びに使う）。"""
    return tuple(_PROVIDERS.values())


def keys() -> tuple[str, ...]:
    return tuple(_PROVIDERS)


__all__ = ['TtsProvider', 'all_providers', 'get', 'keys', 'register']
