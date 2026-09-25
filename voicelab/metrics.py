"""計測結果の記録と、表の出力。

1 往復（質問 1 つ → 回答 1 つ）を 1 行として CSV に積む。構成（A / B / C と、B の TTS を
差し替えた各社）ごとに中央値を出して Markdown の表にする。数字の意味は README の「計測の定義」。
表は ``results/`` 直下に 1 つ。録音・書き起こしは ``results/<会社>/<モデル>/`` に分かれる。
"""

import csv
import statistics
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from . import tts_provider
from .config import RESULTS_DIR

RUNS_CSV = RESULTS_DIR / 'runs.csv'
REPORT_MD = RESULTS_DIR / 'report.md'

#: 組み込みの構成と TTS の綴り。CSV の ``path`` 列にそのまま入る値なので、**一度書いた綴りは変えない**
#: （変えると過去の行が読めなくなる）。C・D を足しても A・B の行はそのまま読める。
PATH_AGENTS = 'agents'  # A: Agents Platform（ノートは手元、道具で引く）
PATH_CUSTOM = 'custom'  # B: 自前構成（ノートは手元、先に引いて渡す）
PATH_KB = 'kb'  # C: Knowledge Base（ノートを ElevenLabs に預け、向こうの RAG で引く）
#: B（Deepgram）: B の TTS だけを Deepgram Aura-2 に替えたもの。
#: ElevenLabs を使わないので ``credits`` 列（ElevenLabs のクレジット）は 0 になる。
#: 費用はドルで ``note`` 列と書き起こしに書く。
PATH_DEEPGRAM = 'deepgram'
#: 組み込みの 4 つ。B の TTS を他社に差し替えた構成（OpenAI / Google …）は
#: :mod:`voicelab.tts_provider` に登録された ``key`` が加わる（:func:`known_paths`）。
PATHS = (PATH_AGENTS, PATH_CUSTOM, PATH_KB, PATH_DEEPGRAM)

#: 組み込みの見出し。登録されたプロバイダの分は :func:`path_label` が登録簿から引く。
BUILTIN_LABELS = {
    PATH_AGENTS: 'A: Agents Platform',
    PATH_CUSTOM: 'B: 自前構成（ElevenLabs）',
    PATH_KB: 'C: Knowledge Base',
    PATH_DEEPGRAM: 'B: 自前構成（Deepgram）',
}


def known_paths() -> tuple[str, ...]:
    """CSV の ``path`` 列に入りうる値。組み込み + 登録済みプロバイダ（順は登録順、重複なし）。"""
    paths = list(PATHS)
    for key in tts_provider.keys():
        if key not in paths:
            paths.append(key)
    return tuple(paths)


def path_label(path: str) -> str:
    """表の見出し。組み込みは固定、登録済みプロバイダは登録簿の ``label``、どちらでもなければそのまま。"""
    if path in BUILTIN_LABELS:
        return BUILTIN_LABELS[path]
    try:
        return tts_provider.get(path).label
    except KeyError:
        return path


@dataclass
class Run:
    """1 往復の記録。

    :ivar scenario_id: ``scenarios/questions.json`` の id。
    :ivar path: ``agents`` / ``custom`` / ``kb`` / ``deepgram``。
    :ivar first_audio_ms: 話し終わってから最初の音が出るまで。体感の遅延はこれ。
    :ivar reply_done_ms: 話し終わってから回答が言い終わるまで。
    :ivar credits: この往復で消費した ElevenLabs のクレジット（会話の ``cost``、無ければ残高の差）。
        ElevenLabs を使わない構成（Deepgram / OpenAI / Google …）は 0。
    :ivar usd: この往復のドル（見積り）。会社をまたいで並べるための共通の単位。
        ElevenLabs の B は API の定価（Flash / Turbo $0.05 / 1,000 字）で換算、A・C は会話課金で
        文字数から出せないので 0 のまま（表では「—」）。
    :ivar model: TTS のモデル（声）。``results/<会社>/<モデル>/`` のモデルの段と同じ綴り。
        A・C は ``agents-platform`` / ``knowledge-base``。2026-09-25 より前の行は空。
    :ivar correct: 期待したノートを根拠に答えたか（人が判定して入れる）。
    :ivar note: 気づいたこと。割り込まれた、ツールを呼ばなかった、など。
    """

    scenario_id: str
    path: str
    first_audio_ms: int
    reply_done_ms: int
    credits: int
    correct: bool | None = None
    note: str = ''
    taken_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    usd: float = 0.0
    model: str = ''


def append_run(run: Run, path: Path = RUNS_CSV) -> None:
    """1 行追記する。ファイルが無ければ見出し付きで作る。"""
    if run.path not in known_paths():
        raise ValueError(f'path は {known_paths()} のどれか: {run.path!r}')
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Run)])
        if new_file:
            writer.writeheader()
        writer.writerow(asdict(run))


def load_runs(path: Path = RUNS_CSV) -> list[Run]:
    """CSV を読んで :class:`Run` の一覧にする（無ければ空）。"""
    if not path.exists():
        return []
    runs: list[Run] = []
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            runs.append(
                Run(
                    scenario_id=row['scenario_id'],
                    path=row['path'],
                    first_audio_ms=int(row['first_audio_ms']),
                    reply_done_ms=int(row['reply_done_ms']),
                    credits=int(row['credits']),
                    correct=_parse_bool(row.get('correct', '')),
                    note=row.get('note', ''),
                    taken_at=row.get('taken_at', ''),
                    usd=float(row.get('usd') or 0),
                    model=row.get('model') or '',
                )
            )
    return runs


def _parse_bool(value: str) -> bool | None:
    if value in ('True', 'true', '1'):
        return True
    if value in ('False', 'false', '0'):
        return False
    return None


def summarize(runs: list[Run]) -> dict[str, dict[str, float | int | str]]:
    """構成ごとの要約（件数・遅延の中央値・費用の合計と平均・正答率）。"""
    summary: dict[str, dict[str, float | int | str]] = {}
    for path in known_paths():
        rows = [r for r in runs if r.path == path]
        if not rows:
            continue
        judged = [r for r in rows if r.correct is not None]
        summary[path] = {
            'runs': len(rows),
            'first_audio_ms_median': statistics.median(r.first_audio_ms for r in rows),
            'reply_done_ms_median': statistics.median(r.reply_done_ms for r in rows),
            'credits_total': sum(r.credits for r in rows),
            'credits_mean': round(statistics.mean(r.credits for r in rows)),
            'usd_total': round(sum(r.usd for r in rows), 5),
            'usd_mean': round(statistics.mean(r.usd for r in rows), 5),
            'correct_rate': (
                f'{sum(1 for r in judged if r.correct)}/{len(judged)}' if judged else '未判定'
            ),
        }
    return summary


def render_report(runs: list[Run]) -> str:
    """要約と全行を Markdown にする。"""
    lines = ['# 計測結果', '']
    summary = summarize(runs)
    if not summary:
        lines.append('まだ記録がありません。')
        return '\n'.join(lines) + '\n'

    lines += [
        '| 構成 | 往復数 | 最初の音まで（中央値 ms） | 言い終わるまで（中央値 ms） '
        '| クレジット合計 | 1 往復あたり | ドル合計（見積り） | 1 往復あたり $ | 正答 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for path, row in summary.items():
        usd_total = f'{row["usd_total"]:.4f}' if row['usd_total'] else '—'
        usd_mean = f'{row["usd_mean"]:.4f}' if row['usd_mean'] else '—'
        lines.append(
            f'| {path_label(path)} | {row["runs"]} | {row["first_audio_ms_median"]:.0f} '
            f'| {row["reply_done_ms_median"]:.0f} | {row["credits_total"]:,} '
            f'| {row["credits_mean"]:,} | {usd_total} | {usd_mean} | {row["correct_rate"]} |'
        )
    lines += [
        '',
        'ドルは各社の定価からの見積り（文字数課金の社は文字数、OpenAI は音声の長さから。A・C は会話課金なので出せず「—」）。'
        '正はそれぞれの残高の差。',
    ]

    lines += ['', '## 全記録', '',
              '| 日時 (UTC) | 質問 | 構成 | モデル | 最初の音 ms | 言い終わり ms | クレジット | $ | 正答 | メモ |',
              '|---|---|---|---|---:|---:|---:|---:|---|---|']
    for r in runs:
        correct = '' if r.correct is None else ('○' if r.correct else '×')
        usd = f'{r.usd:.5f}' if r.usd else ''
        lines.append(
            f'| {r.taken_at[:16].replace("T", " ")} | {r.scenario_id} | {r.path} | {r.model} '
            f'| {r.first_audio_ms} | {r.reply_done_ms} | {r.credits} | {usd} | {correct} | {r.note} |'
        )
    return '\n'.join(lines) + '\n'


def write_report(runs: list[Run], path: Path = REPORT_MD) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(runs), encoding='utf-8')
    return path


if __name__ == '__main__':
    # python -m voicelab.metrics
    # 記録済みの CSV から表を作って表示する（課金なし）。
    print(render_report(load_runs()))
