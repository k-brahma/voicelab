"""2026-09-25 の置き場の変更に合わせて、既存の ``results/`` を並べ替える（1 回きり）。

旧: ``results/{audio,raw,transcripts}/`` に全構成が混在
新: ``results/<会社>/<モデル>/{audio,raw,transcripts}/``

判定の材料:

- 書き起こし（``*.txt``）は「構成:」の行から会社とモデルを読む。A（``agents``）と C（``kb``）は
  モデルの段に ``agents-platform`` / ``knowledge-base`` を入れる
- 録音（``*.wav``）は同じ ``<質問>_<構成>_<時刻>`` の書き起こしと**同じ場所**へ。対になる書き起こしが
  無いもの（声やモデルの聞き比べ ``voice-sample_*`` など）は ``elevenlabs/samples/audio/``
- 会話の生 JSON は ``agent_id`` が ``.env`` の ``ELEVENLABS_KB_AGENT_ID`` なら ``knowledge-base``、
  それ以外は ``agents-platform``
- ``runs.csv`` には ``usd`` と ``model`` の列を足す。``model`` は同じ時刻の書き起こしから、
  ``usd`` は ElevenLabs の B は ``credits`` から（1 クレジット = 2 文字、$0.05 / 1,000 字）、
  Deepgram は ``note`` 末尾の ``$…`` から。A・C は 0

``--dry-run`` で何をするかだけ出す。実行しても元のフォルダは空になるだけで消さない。
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicelab.config import (  # noqa: E402
    COMPANY_ELEVENLABS,
    MODEL_AGENTS_PLATFORM,
    MODEL_KNOWLEDGE_BASE,
    MODEL_SAMPLES,
    RESULTS_DIR,
    load_env,
    result_dirs,
)

OLD_AUDIO = RESULTS_DIR / 'audio'
OLD_RAW = RESULTS_DIR / 'raw'
OLD_TRANSCRIPTS = RESULTS_DIR / 'transcripts'
RUNS_CSV = RESULTS_DIR / 'runs.csv'

NAME_RE = re.compile(r'^(?P<scenario>.+)_(?P<path>agents|custom|kb|deepgram)_(?P<stamp>\d{8}T\d{6}Z)$')
CONFIG_RE = re.compile(r'^構成: .*?→ (?P<company>ElevenLabs|Deepgram) (?P<model>[^\s）]+)', re.M)

USD_PER_CREDIT_CUSTOM = 0.0001  # 1 クレジット = 2 文字（0.5 クレジット/字）× $0.05 / 1,000 字


def classify_transcript(path: Path) -> tuple[str, str]:
    """書き起こし 1 本の (会社, モデル)。"""
    m = NAME_RE.match(path.stem)
    if not m:
        raise ValueError(f'名前の形が違う: {path.name}')
    kind = m['path']
    if kind == 'agents':
        return COMPANY_ELEVENLABS, MODEL_AGENTS_PLATFORM
    if kind == 'kb':
        return COMPANY_ELEVENLABS, MODEL_KNOWLEDGE_BASE
    text = path.read_text(encoding='utf-8')
    c = CONFIG_RE.search(text)
    if not c:
        raise ValueError(f'「構成:」の行が読めない: {path.name}')
    return c['company'].lower(), c['model']


def plan(env: dict[str, str]) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    by_key: dict[str, tuple[str, str]] = {}

    for t in sorted(OLD_TRANSCRIPTS.glob('*.txt')) if OLD_TRANSCRIPTS.exists() else []:
        company, model = classify_transcript(t)
        by_key[t.stem] = (company, model)
        moves.append((t, result_dirs(company, model).transcripts / t.name))

    for w in sorted(OLD_AUDIO.glob('*.wav')) if OLD_AUDIO.exists() else []:
        if w.stem in by_key:
            company, model = by_key[w.stem]
        else:
            company, model = COMPANY_ELEVENLABS, MODEL_SAMPLES
        moves.append((w, result_dirs(company, model).audio / w.name))

    kb_agent = env.get('ELEVENLABS_KB_AGENT_ID')
    for j in sorted(OLD_RAW.glob('*.json')) if OLD_RAW.exists() else []:
        agent_id = json.loads(j.read_text(encoding='utf-8')).get('agent_id')
        model = MODEL_KNOWLEDGE_BASE if kb_agent and agent_id == kb_agent else MODEL_AGENTS_PLATFORM
        moves.append((j, result_dirs(COMPANY_ELEVENLABS, model).raw / j.name))
    return moves


def migrate_csv(dry_run: bool) -> tuple[int, int]:
    """``usd`` と ``model`` の列を足す。既にあれば何もしない。戻り値は (行数, model を埋めた行数)。"""
    with RUNS_CSV.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if 'usd' in fieldnames and 'model' in fieldnames:
        return len(rows), sum(1 for r in rows if r.get('model'))

    # 時刻 → モデル（書き起こしの名前と「構成:」行から）。書き起こしは移行前でも後でも探す
    stamp_to_model: dict[str, str] = {}
    for base in [OLD_TRANSCRIPTS, *RESULTS_DIR.glob('*/*/transcripts')]:
        for t in base.glob('*.txt') if base.exists() else []:
            m = NAME_RE.match(t.stem)
            if not m:
                continue
            try:
                stamp_to_model[(m['path'], m['stamp'])] = classify_transcript(t)[1]
            except ValueError:
                continue

    filled = 0
    for r in rows:
        r.setdefault('usd', '')
        r.setdefault('model', '')
        stamp = r['taken_at'].replace('-', '').replace(':', '')[:15] + 'Z'  # 2026-09-11T02:26:52.123+00:00 → 20260911T022652Z
        model = stamp_to_model.get((r['path'], stamp), '')
        if not model and r['path'] == 'agents':
            model = MODEL_AGENTS_PLATFORM
        if not model and r['path'] == 'kb':
            model = MODEL_KNOWLEDGE_BASE
        if model:
            filled += 1
        r['model'] = r['model'] or model
        if not r['usd']:
            if r['path'] == 'custom':
                r['usd'] = f'{int(r["credits"]) * USD_PER_CREDIT_CUSTOM:.6f}'
            elif r['path'] == 'deepgram':
                found = re.search(r'\$(\d+\.\d+)\s*$', r['note'])
                r['usd'] = found.group(1) if found else '0'
            else:
                r['usd'] = '0'
    new_fields = [f for f in fieldnames if f not in ('usd', 'model')] + ['usd', 'model']
    if not dry_run:
        with RUNS_CSV.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=new_fields)
            writer.writeheader()
            writer.writerows(rows)
    return len(rows), filled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    env = load_env()
    moves = plan(env)
    for src, dst in moves:
        print(f'{src.relative_to(RESULTS_DIR)}  →  {dst.relative_to(RESULTS_DIR)}')
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    print(f'{len(moves)} 件{"（dry-run）" if args.dry_run else ""}')

    rows, filled = migrate_csv(args.dry_run)
    print(f'runs.csv: {rows} 行に usd / model を追加（model を埋めた行 {filled}）{"（dry-run）" if args.dry_run else ""}')

    if not args.dry_run:
        for old in (OLD_AUDIO, OLD_RAW, OLD_TRANSCRIPTS):
            if old.exists() and not any(old.iterdir()):
                old.rmdir()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
