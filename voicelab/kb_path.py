"""C: Knowledge Base で 1 往復して、遅延と費用を記録する。

**責務はここだけ**。「質問を 1 つ投げて、返事を最後まで受け取り、時刻とクレジットを残す」。
質問の選び方、残高の見張り、表の更新は :mod:`voicelab.cli` の仕事。

A（:mod:`voicelab.agents_path`）との違いは**たった 2 つ**。

- ``client_tools`` を渡さない。C の Agent は道具を 1 つも持たない
- 使う Agent が ``ELEVENLABS_KB_AGENT_ID``（ノート本文を預けてある方）

ほかは全部 A と同じものを**再利用する**（:class:`~voicelab.agents_path.MeasuringAudioInterface`、
``audio_window_ms``、``reply_state``、``save_audio_file``、``save_transcript``、
接続待ちと返答待ち、会話メタデータの取得）。計測の道具を写し取ると、直したときに片方だけ
直り、A と C の数字が比べられなくなる。だからここには計測のコードを 1 行も置かない。

C だけが記録するもの: 会話メタデータの ``rag_usage``。**ElevenLabs 側の RAG が実際に
引いたかどうかの証拠**で、これが空なら「速かったのは何も引かなかったから」を疑う。
"""

import time

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation

from .agents_path import (
    AgentsPathError,
    Capture,
    MeasuringAudioInterface,
    audio_window_ms,
    build_note,
    conversation_cost,
    fetch_conversation_metadata,
    render_transcript,
    save_audio_file,
    save_transcript,
    utc_stamp,
    wait_for_connection,
    wait_for_reply,
)
from .config import COMPANY_ELEVENLABS, MODEL_KNOWLEDGE_BASE, load_env, require, result_dirs
from .kb_setup import KB_AGENT_ID_KEY, KB_AGENT_NAME, RAG_MODEL, corpus_documents, fetch_agent_summary
from .metrics import PATH_KB, Run


class KbPathError(AgentsPathError):
    """会話を始められなかった、または続けられなかった。

    :class:`voicelab.agents_path.AgentsPathError` を継いでいるのは、CLI の受け口を
    増やさずに A と同じ扱いにするため。
    """


def rag_summary(rag_usage: dict | None) -> str:
    """``rag_usage`` を CSV の 1 列に収まる一言にする純粋関数。

    形が版によって違う（``{"usage": ...}`` だったり件数の内訳だったり）ので、
    **深追いせずに「あったか / 無かったか」と原文の短い断片**だけを残す。
    正確な中身は書き起こし（``results/transcripts/``）に丸ごと書いてある。
    """
    if not rag_usage:
        return 'RAG=記録なし'
    text = str(rag_usage).replace('\n', ' ')
    return f'RAG={text[:40]}'


def describe_dry_run(
    scenario: dict, env: dict[str, str] | None = None, summary: dict | None = None
) -> str:
    """接続せずに、C の Agent が「知識を預かっていて道具を持たない」状態かを見せる。

    A の ``--dry-run`` は手元の検索結果を見せるが、**C に手元の検索は無い**。代わりに
    確かめたいのは次の 3 つで、どれも会話を開かずに分かる（クレジットは 1 も使わない）。

    - Agent が Knowledge Base を持っているか（何本か）
    - 索引の埋め込みモデルが日本語向き（:data:`~voicelab.kb_setup.RAG_MODEL`）か
    - 道具が空か（空でなければ、それは C ではなく A の設定）

    :param summary: :func:`voicelab.kb_setup.summarize_agent` の結果。省くと API を
        1 回読む（無料）。テストはここにそのまま渡してネットワークを使わない。
    """
    env = load_env() if env is None else env
    agent_id = env.get(KB_AGENT_ID_KEY, '')
    api_key = env.get('ELEVENLABS_API_KEY', '')
    lines = [
        f'[dry-run] {scenario["id"]}: {scenario["text"]}',
        f'  Agent {KB_AGENT_NAME}: {agent_id or "（未設定。setup-kb を先に実行）"}',
        f'  声/モデル: {env.get("ELEVENLABS_VOICE_ID", "")} / {env.get("ELEVENLABS_MODEL_ID", "")}',
        f'  期待するノート: {scenario.get("expected_note")}',
        f'  手元の corpus: {len(corpus_documents())} 本（C はこれを ElevenLabs に預けてある）',
    ]
    if summary is None and agent_id and api_key:
        summary = fetch_agent_summary(api_key, agent_id)
    if summary is None:
        lines.append('  Agent の設定: 読めません（鍵か agent id が未設定）')
    else:
        documents = summary.get('documents') or []
        tool_ids = summary.get('tool_ids') or []
        tools_line = (
            'なし（C の想定どおり）'
            if not tool_ids
            else f'{len(tool_ids)} 個あります（C では空のはず）'
        )
        lines += [
            f'  預けてある文書: {len(documents)} 本  {" / ".join(documents) or "（無し）"}',
            f'  RAG: {"有効" if summary.get("rag_enabled") else "無効"}'
            f'  埋め込み={summary.get("embedding_model") or "（未設定）"}'
            f'  最大チャンク={summary.get("max_chunks")}',
            f'  道具: {tools_line}',
        ]
        if summary.get('embedding_model') and summary['embedding_model'] != RAG_MODEL:
            lines.append(f'  ※ 埋め込みが {RAG_MODEL} ではありません。日本語では引けません')
        if not documents:
            lines.append('  ※ 文書が紐付いていません。setup-kb を実行してください')
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

    **クレジットを消費する。** 呼ぶ前に残高を確かめること（``voicelab run kb`` がやる）。
    流れは A と同じで、違うのは道具を渡さないことと Agent の id だけ。

    :raises KbPathError: 接続できなかった。
    """
    env = load_env() if env is None else env
    api_key = require('ELEVENLABS_API_KEY', env)
    agent_id = require(KB_AGENT_ID_KEY, env)

    capture = Capture()
    audio = MeasuringAudioInterface(send_silence=send_silence)

    conversation = Conversation(
        ElevenLabs(api_key=api_key),
        agent_id,
        requires_auth=True,
        audio_interface=audio,
        # client_tools は渡さない。C の Agent は道具を持たず、ノートは ElevenLabs 側の
        # RAG が引く。ここで渡すと、比べたかった「ツール往復の有無」が消える
        callback_agent_response=capture.responses.append,
        callback_user_transcript=capture.transcripts.append,
    )

    conversation.start_session()
    try:
        try:
            wait_for_connection(conversation)
        except AgentsPathError as exc:
            raise KbPathError(
                f'{exc} （C では {KB_AGENT_ID_KEY} を使います。setup-kb を実行しましたか）'
            ) from exc
        t0 = time.perf_counter()
        conversation.send_user_message(scenario['text'])
        state = wait_for_reply(t0, audio)
    finally:
        conversation.end_session()
        conversation_id = conversation.wait_for_session_end()

    first_audio_ms, reply_done_ms = audio_window_ms(t0, audio.times)
    metadata = (
        fetch_conversation_metadata(api_key, conversation_id) if conversation_id else None
    )
    credits_used, duration_secs = conversation_cost(metadata)
    rag_usage = (metadata or {}).get('rag_usage')

    stamp = utc_stamp()
    dirs = result_dirs(COMPANY_ELEVENLABS, MODEL_KNOWLEDGE_BASE)
    if save_audio and audio.chunks:
        save_audio_file(audio.pcm(), scenario['id'], stamp, dirs.audio, label=PATH_KB)
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
            # 会話が取れなくても節は出す。「記録なし」も結果のうち
            rag_usage=rag_usage if rag_usage is not None else {},
        ),
        scenario['id'],
        stamp,
        dirs.transcripts,
        label=PATH_KB,
    )

    return Run(
        scenario_id=scenario['id'],
        path=PATH_KB,
        model=MODEL_KNOWLEDGE_BASE,
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
            lead=f'道具なし {rag_summary(rag_usage)}',
        ),
    )


if __name__ == '__main__':
    # python -m voicelab.kb_path rollback
    # **クレジットを消費する。** 残高の見張りは cli 側にあるので、ここには無い。
    import sys

    from .config import find_scenario

    print(run_scenario(find_scenario(sys.argv[1] if len(sys.argv) > 1 else 'rollback')))
