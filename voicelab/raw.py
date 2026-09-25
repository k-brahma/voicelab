"""会話の**生の JSON** を丸ごと落として残す。

``agents_path.fetch_conversation_cost()`` は同じ応答から費用と通話秒数だけを取り出して
捨てている。だが元の JSON には、整形した表には現れないものが入っている。

- ``transcript[].conversation_turn_metrics`` … ターンごとの内訳（TTS の初音まで、など）
- ``transcript[].tool_calls`` / ``tool_results`` … 道具に渡した引数と戻り値
- ``transcript[].llm_usage`` … LLM のトークン
- ``metadata.rag_usage`` … ElevenLabs 内部の RAG が使われたか（C の構成で効く）
- ``metadata.charging`` / ``cost`` … 課金の内訳

会話は ElevenLabs 側にも残っているので、**後からでも取り直せる**。このモジュールは
「一覧を引く」「1 件を丸ごと取る」「ファイルに落とす」の 3 つだけを担い、
解釈（表にする、数字を比べる）は :mod:`voicelab.metrics` の仕事とする。
"""

import json
from pathlib import Path

import httpx

from .config import (
    COMPANY_ELEVENLABS,
    MODEL_AGENTS_PLATFORM,
    MODEL_KNOWLEDGE_BASE,
    RESULTS_DIR,
    load_env,
    result_dirs,
)

API_BASE = 'https://api.elevenlabs.io/v1'
TIMEOUT_SECONDS = 60

#: 旧: 落とし先（2026-09-25 まで）。今は :func:`raw_dir_for` が Agent ごとに決める。
RAW_DIR = RESULTS_DIR / 'raw'


def raw_dir_for(agent_id: str | None, env: dict[str, str] | None = None) -> Path:
    """会話の JSON を置く ``raw/``。Agent が C（Knowledge Base）のものなら ``knowledge-base``、それ以外は ``agents-platform``。

    1 会話 1 ファイル（会話 id がそのままファイル名）。会社の段は ElevenLabs 固定
    （会話の生データがあるのは Agents Platform だけ）。
    """
    env = load_env() if env is None else env
    model = MODEL_KNOWLEDGE_BASE if agent_id and agent_id == env.get('ELEVENLABS_KB_AGENT_ID') else MODEL_AGENTS_PLATFORM
    return result_dirs(COMPANY_ELEVENLABS, model).raw

#: 一覧を引くときの 1 ページの件数。
PAGE_SIZE = 100


class RawError(RuntimeError):
    """会話を取れなかった。"""


def _get(client: httpx.Client, path: str, **params) -> dict:
    try:
        response = client.get(f'{API_BASE}{path}', params=params or None)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise RawError(f'ElevenLabs が HTTP {exc.response.status_code} を返しました') from exc
    except httpx.HTTPError as exc:
        raise RawError('ElevenLabs に接続できませんでした') from exc
    except ValueError as exc:
        raise RawError('応答が JSON ではありません') from exc


def _client(api_key: str) -> httpx.Client:
    return httpx.Client(headers={'xi-api-key': api_key}, timeout=TIMEOUT_SECONDS)


def list_conversations(api_key: str, agent_id: str | None = None) -> list[dict]:
    """会話の一覧を全ページ引く（新しい順）。

    :param agent_id: 指定するとその Agent の会話だけ。``None`` なら全部。
    """
    items: list[dict] = []
    cursor: str | None = None
    with _client(api_key) as client:
        while True:
            params = {'page_size': PAGE_SIZE}
            if agent_id:
                params['agent_id'] = agent_id
            if cursor:
                params['cursor'] = cursor
            payload = _get(client, '/convai/conversations', **params)
            items.extend(payload.get('conversations') or [])
            cursor = payload.get('next_cursor')
            if not payload.get('has_more') or not cursor:
                return items


def fetch_conversation(api_key: str, conversation_id: str) -> dict:
    """1 会話の詳細を**丸ごと**返す（間引かない）。"""
    with _client(api_key) as client:
        return _get(client, f'/convai/conversations/{conversation_id}')


def save(payload: dict, conversation_id: str, directory: Path | None = None) -> Path:
    """生の JSON をそのまま書き出す。既にあれば上書きする。

    ``directory`` を省くと :func:`raw_dir_for` が ``payload['agent_id']`` から決める。

    ``ensure_ascii=False`` は日本語を読める形で残すため。``indent=2`` は
    ``git diff`` と目視のため（1 行 JSON だと差分が読めない）。
    """
    directory = raw_dir_for(payload.get('agent_id')) if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{conversation_id}.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def backfill(api_key: str, agent_ids: list[str] | None = None, *, skip_existing: bool = True) -> list[Path]:
    """ElevenLabs に残っている会話を全部落とす。

    既に落としたものを飛ばすのは、``status`` が ``done`` になった後の内容は
    変わらないため（取り直しても同じ）。

    :param agent_ids: 対象の Agent。``None`` なら全会話。
    :param skip_existing: 既にファイルがあるものを飛ばすか。
    """
    targets: list[dict] = []
    if agent_ids:
        for agent_id in agent_ids:
            targets.extend(list_conversations(api_key, agent_id))
    else:
        targets = list_conversations(api_key)

    saved: list[Path] = []
    with _client(api_key) as client:
        for item in targets:
            conversation_id = item.get('conversation_id')
            if not conversation_id:
                continue
            path = raw_dir_for(item.get('agent_id')) / f'{conversation_id}.json'
            if skip_existing and path.exists():
                continue
            payload = _get(client, f'/convai/conversations/{conversation_id}')
            saved.append(save(payload, conversation_id))
    return saved


def summarize_file(path: Path) -> str:
    """落とした JSON の 1 行要約（中身を見ずに当たりを付けるため）。"""
    payload = json.loads(path.read_text(encoding='utf-8'))
    meta = payload.get('metadata') or {}
    turns = payload.get('transcript') or []
    tools = sum(len(t.get('tool_calls') or []) for t in turns)
    rag = 'あり' if (meta.get('rag_usage') or {}) else 'なし'
    return (
        f'{path.stem}  {meta.get("call_duration_secs", "?"):>3}s  '
        f'cost={meta.get("cost", "?"):>4}  ターン{len(turns):>2}  道具{tools}  RAG={rag}'
    )


if __name__ == '__main__':
    # python -m voicelab.raw              … 残っている会話を全部落とす
    # python -m voicelab.raw <会話 id>    … 1 件だけ落として要約を出す
    # 課金なし（読むだけ）。
    import sys

    from .config import load_env, require

    env = load_env()
    key = require('ELEVENLABS_API_KEY', env)

    if len(sys.argv) > 1:
        cid = sys.argv[1]
        print(summarize_file(save(fetch_conversation(key, cid), cid)))
    else:
        paths = backfill(key)
        print(f'{len(paths)} 件を落とした → {RAW_DIR}')
        for p in sorted(RAW_DIR.glob('*.json')):
            print('  ' + summarize_file(p))
