"""`.env` の読み込み。

外部ライブラリを増やさないため、`KEY=VALUE` だけを見る素朴な実装にしている。
骨組み（credits / report）を標準ライブラリだけで動かせる状態を保つのが狙い。
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / '.env'

#: 結果を書き出す先。CSV と Markdown の表を置く。
RESULTS_DIR = ROOT / 'results'

#: 題材データ（架空のノート）。実際の vault は使わない。
CORPUS_DIR = ROOT / 'corpus'

#: 質問集。
SCENARIOS_PATH = ROOT / 'scenarios' / 'questions.json'

#: Agent に持たせる指示。コードから切り離してあるのは、文言を直したときに
#: コードを触らず `setup-agent` の再実行だけで反映させたいため。
AGENT_PROMPT_PATH = ROOT / 'prompts' / 'agent_system.txt'

#: C（Knowledge Base）に持たせる指示。A との違いは**冒頭の 2 行だけ**で、
#: 「必ず search_notes を呼ぶ」が「ノートを参照できる」に変わっている。
#: 残りの行（長さ・出典・記号・数字）を A と同じ文言にしてあるのは、
#: 測った差が prompt の差にならないようにするため。
KB_PROMPT_PATH = ROOT / 'prompts' / 'agent_system_kb.txt'

#: 会社（プロバイダ）とモデルごとの置き場。``results/<会社>/<モデル>/{audio,raw,transcripts}/``。
#:
#: 会社をまたいで比べるようになったので（ElevenLabs / Deepgram / OpenAI / Google …）、
#: 録音・生データ・書き起こしは「どの会社のどのモデルで出したか」で分けて置く。
#: 表（``runs.csv`` / ``report.md``）だけは ``results/`` 直下に 1 つ。
#: ElevenLabs の A（Agents Platform）と C（Knowledge Base）は TTS のモデルを自分で選ばない
#: 構成なので、モデルの段には構成の名前を入れる。
COMPANY_ELEVENLABS = 'elevenlabs'
MODEL_AGENTS_PLATFORM = 'agents-platform'  # A
MODEL_KNOWLEDGE_BASE = 'knowledge-base'  # C
#: 構成の比較ではない、声やモデルの聞き比べ（``voice-sample_*.wav`` など）の置き場。
MODEL_SAMPLES = 'samples'


def safe_dir_name(name: str) -> str:
    """モデル名をフォルダ名にする。``/`` と ``:`` だけ ``_`` に変える（他はそのまま）。"""
    return name.replace('/', '_').replace(':', '_').strip() or 'unknown'


class ResultDirs:
    """1 つの会社・モデルの ``audio`` / ``raw`` / ``transcripts``。作るのは書くときで、ここでは作らない。"""

    def __init__(self, company: str, model: str, root: Path = RESULTS_DIR):
        self.company = company
        self.model = safe_dir_name(model)
        self.base = root / company / self.model
        self.audio = self.base / 'audio'
        self.raw = self.base / 'raw'
        self.transcripts = self.base / 'transcripts'

    def __repr__(self) -> str:
        return f'ResultDirs({self.company!r}, {self.model!r})'


def result_dirs(company: str, model: str) -> ResultDirs:
    """``results/<会社>/<モデル>/`` の各フォルダ。録音と書き起こしを残すときはこれで場所を決める。"""
    return ResultDirs(company, model)


#: 旧: 会話の録音の置き場（2026-09-25 まで。会社・モデル別に分ける前）。移行スクリプトだけが見る。
AUDIO_DIR = RESULTS_DIR / 'audio'

#: 旧: 書き起こしの置き場（同上）。
TRANSCRIPTS_DIR = RESULTS_DIR / 'transcripts'

#: 環境変数からの上書きを許すキー。`.env` に書かれていなくても拾う。
ENV_KEYS = (
    'ELEVENLABS_API_KEY',
    'ELEVENLABS_AGENT_ID',
    # C（Knowledge Base）の Agent。`setup-kb` が書く。A の Agent とは別物で、
    # 道具を持たない代わりにノート本文を ElevenLabs 側に預けてある
    'ELEVENLABS_KB_AGENT_ID',
    'ELEVENLABS_VOICE_ID',
    'ELEVENLABS_MODEL_ID',
    'VOICELAB_LLM',
    'VOICELAB_TTS_MODEL',
    # B（自前構成）が LLM を自分で呼ぶのに要る。A は ElevenLabs 側で同じモデルを
    # 動かすので鍵は要らない。ここが A と B の費用の出方の違いでもある。
    'GEMINI_API_KEY',
    # B の TTS を他社に差し替えたときの鍵とモデル。登録は各 *_path モジュール
    'DEEPGRAM_API_KEY',
    'DEEPGRAM_TTS_MODEL',
    'OPENAI_API_KEY',
    'OPENAI_TTS_MODEL',
    'OPENAI_TTS_VOICE',
    'GOOGLE_TTS_API_KEY',
    'GOOGLE_TTS_VOICE',
)

#: Agent の応答生成に使う LLM の既定。`.env` の `VOICELAB_LLM` で変えられる。
DEFAULT_LLM = 'gemini-3.6-flash'

#: B（Deepgram）の TTS モデル（Deepgram では声とモデルが 1 つの id）。`.env` の `DEEPGRAM_TTS_MODEL` で変えられる。
#:
#: 2026-09-25 に ``GET /v1/models`` で確かめたところ、日本語（``ja`` / ``ja-JP``）を持つ TTS は
#: Aura-2 の 5 声（``aura-2-ama-ja`` / ``ebisu`` / ``fujin`` / ``izanami`` / ``uzume``）だけだった。
#: B の声（Jessica）と同じ女性で、用途の札（会話・窓口・面接・IVR）が一番広い Izanami にしている。
DEFAULT_DEEPGRAM_TTS_MODEL = 'aura-2-izanami-ja'


class ConfigError(RuntimeError):
    """設定が足りない。"""


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """`.env` を読んで辞書で返す（無ければ空）。

    既に環境変数にある値を優先する。CI や一時的な上書きから使えるようにするため。
    """
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in list(values) + list(ENV_KEYS):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict]:
    """質問集を読む。A と B が**同じ質問を同じ順で**流すための唯一の入口。"""
    return json.loads(path.read_text(encoding='utf-8'))['questions']


def find_scenario(scenario_id: str, path: Path = SCENARIOS_PATH) -> dict:
    """id で 1 問だけ取る。

    :raises ConfigError: その id が質問集に無い。
    """
    for item in load_scenarios(path):
        if item['id'] == scenario_id:
            return item
    known = ', '.join(item['id'] for item in load_scenarios(path))
    raise ConfigError(f'{scenario_id!r} という質問はありません。あるのは: {known}')


def require(key: str, env: dict[str, str] | None = None) -> str:
    """必須の設定を取る。無ければ何を書けばよいかを言って止まる。

    :raises ConfigError: 値が空、または未設定。
    """
    env = load_env() if env is None else env
    value = env.get(key, '')
    if not value:
        raise ConfigError(f'{key} が未設定です。.env に書いてください（.env.example を参照）。')
    return value
