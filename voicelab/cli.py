"""コマンドラインの入口。

- ``credits`` … 今の残高を表示（ElevenLabs のクレジットと、鍵のある各社のドル。課金されない）
- ``scenarios`` … 質問集を表示
- ``report`` … 記録済みの結果から表を作る
- ``setup-agent`` … A の Agent とツールを作る／更新する（課金されない）
- ``setup-kb`` … C の Knowledge Base と Agent を作る／更新する（課金されない）
- ``run agents`` … A で質問集を流して計測する（**クレジットを消費する**）
- ``run custom [--tts 社名]`` … B で質問集を流して計測する。TTS は ``--tts`` で差し替える
  （既定は ``elevenlabs``。``deepgram`` / ``openai`` / ``google`` …。**その社の残高と Gemini の
  トークンを消費する**）
- ``run kb`` … C で質問集を流して計測する（**クレジットを消費する**）

**この層の責務は「何回・どの順で流すか」と「流してよいかの判断」**。
1 往復の中身は :mod:`voicelab.agents_path`（A）、:mod:`voicelab.custom_path`（B、ElevenLabs）、
:mod:`voicelab.kb_path`（C）、B の TTS を差し替えた各社は :mod:`voicelab.tts_provider` に
登録されたモジュール（:data:`PROVIDER_MODULES`）にある。
"""

import argparse
import sys

from . import (
    agent_setup,
    agents_path,
    credits,
    custom_path,
    deepgram_path,
    google_path,
    kb_path,
    kb_setup,
    metrics,
    openai_path,
    tts_provider,
)
from .config import ConfigError, find_scenario, load_env, load_scenarios, require

#: B の TTS プロバイダのモジュール。**import された時点で登録される**ので、ここに並べる
#: （並べないと ``--tts`` の選択肢に出ない）。足すときは 1 行追加するだけ。
PROVIDER_MODULES = (custom_path, deepgram_path, openai_path, google_path)

#: ``--tts`` で使える別名。CSV の ``path`` 列の綴り（``custom``）は歴史的なもので、
#: 人が打つときは社名の方が自然なので、社名 → 登録キーに読み替える。
TTS_ALIASES = {'elevenlabs': metrics.PATH_CUSTOM}

#: これを下回っていたら会話を始めない（クレジット）。
#: 2026-09-10 の実測で約 730 クレジット/分。5 問流すなら 2 分前後は見ておく。
MIN_CREDITS = 1500


def tts_choices() -> list[str]:
    """``--tts`` の選択肢。別名（``elevenlabs``）と登録キー（``custom`` / ``deepgram`` …）の両方。"""
    registered = [module.PROVIDER.key for module in PROVIDER_MODULES]
    return list(TTS_ALIASES) + [key for key in registered if key not in TTS_ALIASES.values()]


def resolve_tts(name: str) -> tts_provider.TtsProvider:
    return tts_provider.get(TTS_ALIASES.get(name, name))


def cmd_credits(_args: argparse.Namespace) -> int:
    env = load_env()
    snapshot = credits.read_subscription(require('ELEVENLABS_API_KEY', env))
    print(snapshot.describe())
    for provider in tts_provider.all_providers():
        if provider.uses_elevenlabs_credits or provider.describe_balance is None:
            continue
        if not all(env.get(key) for key in provider.required_env if key != 'GEMINI_API_KEY'):
            continue  # その社を使わない人には出さない
        print(provider.describe_balance(env))
    return 0


def cmd_scenarios(_args: argparse.Namespace) -> int:
    questions = load_scenarios()
    for item in questions:
        print(f'{item["id"]:<16} {item["text"]}   → {item["expected_note"]}')
    print(f'\n{len(questions)} 問。題材: corpus/（架空のノート 3 本）')
    return 0


def cmd_report(_args: argparse.Namespace) -> int:
    runs = metrics.load_runs()
    path = metrics.write_report(runs)
    print(metrics.render_report(runs))
    print(f'→ {path}')
    return 0


def cmd_setup_agent(_args: argparse.Namespace) -> int:
    print(agent_setup.setup().describe())
    return 0


def cmd_setup_kb(args: argparse.Namespace) -> int:
    print(kb_setup.setup(recreate=args.recreate).describe())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.path == metrics.PATH_AGENTS:
        return _run_agents(args)
    if args.path == metrics.PATH_KB:
        return _run_kb(args)
    if args.path == metrics.PATH_CUSTOM:
        return _run_provider(args, resolve_tts(args.tts))
    # ``run deepgram`` のように、登録キーを直接指名した場合（古い呼び方）
    return _run_provider(args, resolve_tts(args.path))


def _run_agents(args: argparse.Namespace) -> int:
    """A を流す。**残高の見張りと、1 問 1 会話の縛りはここで持つ。**"""
    scenarios = [find_scenario(args.scenario)] if args.scenario else load_scenarios()

    if args.dry_run:
        for scenario in scenarios:
            print(agents_path.describe_dry_run(scenario))
            print()
        print('[dry-run] 接続していません。クレジットは消費していません。')
        return 0

    api_key = require('ELEVENLABS_API_KEY')
    before = credits.read_subscription(api_key)
    print(f'実行前 {before.describe()}')
    if before.remaining < MIN_CREDITS and not args.force:
        print(
            f'残り {before.remaining:,} は {MIN_CREDITS:,} を下回っています。'
            ' 会話を始めません（続けるなら --force）。',
            file=sys.stderr,
        )
        return 1

    for scenario in scenarios:
        # 1 問ごとに新しい会話を開く。前の質問の文脈が残ると、ツールを呼ばずに
        # 前の答えを流用することがあり、1 往復の計測にならないため。
        print(f'--- {scenario["id"]}: {scenario["text"]}')
        run = agents_path.run_scenario(scenario, save_audio=not args.no_audio)
        metrics.append_run(run)
        print(
            f'    最初の音 {run.first_audio_ms} ms / 言い終わり {run.reply_done_ms} ms'
            f' / {run.credits} クレジット'
        )
        print(f'    {run.note}')

    after = credits.read_subscription(api_key)
    print(f'実行後 {after.describe()}')
    print(f'消費: {credits.consumed(before, after):,} クレジット（{len(scenarios)} 往復）')
    print(f'→ {metrics.write_report(metrics.load_runs())}')
    print('正誤（correct 列）は音を聞いて results/runs.csv に手で入れる。音は results/elevenlabs/agents-platform/audio/。')
    return 0


def _run_kb(args: argparse.Namespace) -> int:
    """C を流す。**残高の見張りは A と同じ門**（会話なので消費の仕方も A と同じ）。

    A との違いは 1 往復の中身（道具を持たず、ElevenLabs 側の RAG が引く）だけなので、
    ここは ``agents_path`` を ``kb_path`` に読み替えた以外、:func:`_run_agents` と同じにしてある。
    止まる条件や 1 問 1 会話の縛りを構成ごとに変えると、数字が並べられなくなる。
    """
    scenarios = [find_scenario(args.scenario)] if args.scenario else load_scenarios()

    if args.dry_run:
        for scenario in scenarios:
            print(kb_path.describe_dry_run(scenario))
            print()
        print('[dry-run] 会話していません。クレジットは消費していません。')
        return 0

    api_key = require('ELEVENLABS_API_KEY')
    require('ELEVENLABS_KB_AGENT_ID')  # 先に見る。無いまま残高を読んでも進めない

    before = credits.read_subscription(api_key)
    print(f'実行前 {before.describe()}')
    if before.remaining < MIN_CREDITS and not args.force:
        print(
            f'残り {before.remaining:,} は {MIN_CREDITS:,} を下回っています。'
            ' 会話を始めません（続けるなら --force）。',
            file=sys.stderr,
        )
        return 1

    for scenario in scenarios:
        # A と同じく 1 問ごとに新しい会話を開く。前の質問の文脈が残ると 1 往復の計測にならない。
        print(f'--- {scenario["id"]}: {scenario["text"]}')
        run = kb_path.run_scenario(scenario, save_audio=not args.no_audio)
        metrics.append_run(run)
        print(
            f'    最初の音 {run.first_audio_ms} ms / 言い終わり {run.reply_done_ms} ms'
            f' / {run.credits} クレジット'
        )
        print(f'    {run.note}')

    after = credits.read_subscription(api_key)
    print(f'実行後 {after.describe()}')
    print(f'消費: {credits.consumed(before, after):,} クレジット（{len(scenarios)} 往復）')
    print(f'→ {metrics.write_report(metrics.load_runs())}')
    print('正誤（correct 列）は音を聞いて results/runs.csv に手で入れる。音は results/elevenlabs/knowledge-base/audio/。')
    print('RAG が引かれたかは results/elevenlabs/knowledge-base/transcripts/ の rag_usage を見る。')
    return 0


def _run_provider(args: argparse.Namespace, provider: tts_provider.TtsProvider) -> int:
    """B を、TTS プロバイダを 1 つ選んで流す。**鍵の確認と残高の見張りはここで持つ。**

    流し方（1 問ずつ・同じ順・記録と表の更新）はどの社でも同じ。違うのは残高の門だけ。
    ElevenLabs を使う社（B の既定）は A・C と同じ :data:`MIN_CREDITS` の門をくぐらせる
    （B の 1 往復は 100 字で 50 クレジット前後しか使わず、実際にはまず引っかからないが、
    構成ごとに「止まる条件」を変えないため）。他社は残高を読めたら表示するだけで、
    止まる条件にはしない（鍵の権限で読めないことがあり、残高不足は TTS 側が知らせる）。
    """
    scenarios = [find_scenario(args.scenario)] if args.scenario else load_scenarios()

    if args.dry_run:
        for scenario in scenarios:
            print(provider.describe_dry_run(scenario))
            print()
        print(f'[dry-run] 接続していません。{provider.name} の残高も Gemini のトークンも使っていません。')
        return 0

    # 鍵は残高を読む前に確かめる。足りないまま先に進んでも、途中で止まるだけ。
    env = load_env()
    for key in provider.required_env:
        require(key, env)

    before = None
    if provider.uses_elevenlabs_credits:
        before = credits.read_subscription(env['ELEVENLABS_API_KEY'])
        print(f'実行前 {before.describe()}')
        if before.remaining < MIN_CREDITS and not args.force:
            print(
                f'残り {before.remaining:,} は {MIN_CREDITS:,} を下回っています。'
                ' 実行しません（続けるなら --force）。',
                file=sys.stderr,
            )
            return 1
    elif provider.describe_balance is not None:
        print(f'実行前 {provider.describe_balance(env)}')

    for scenario in scenarios:
        print(f'--- {scenario["id"]}: {scenario["text"]}')
        run = provider.run_scenario(scenario, save_audio=not args.no_audio)
        metrics.append_run(run)
        cost = f'{run.credits} {provider.cost_hint}' if provider.uses_elevenlabs_credits else f'${run.usd:.5f}（見積り）'
        print(f'    最初の音 {run.first_audio_ms} ms / 言い終わり {run.reply_done_ms} ms / {cost}')
        print(f'    {run.note}')

    if before is not None:
        after = credits.read_subscription(env['ELEVENLABS_API_KEY'])
        print(f'実行後 {after.describe()}')
        print(
            f'残高の差: {credits.consumed(before, after):,} クレジット（{len(scenarios)} 往復）。'
            ' 反映は遅れるので、見積りと合わないことがある'
        )
    elif provider.describe_balance is not None:
        print(f'実行後 {provider.describe_balance(env)}')
    print(f'→ {metrics.write_report(metrics.load_runs())}')
    print('正誤（correct 列）は音を聞いて results/runs.csv に手で入れる。音は results/<会社>/<モデル>/audio/。')
    for hint in provider.after_run_hints:
        print(hint)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='voicelab', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('credits', help='今の残高を表示').set_defaults(func=cmd_credits)
    sub.add_parser('scenarios', help='質問集を表示').set_defaults(func=cmd_scenarios)
    sub.add_parser('report', help='記録から表を作る').set_defaults(func=cmd_report)
    sub.add_parser(
        'setup-agent', help='A の Agent とツールを作る／更新する（課金されない）'
    ).set_defaults(func=cmd_setup_agent)

    setup_kb = sub.add_parser(
        'setup-kb', help='C の Knowledge Base と Agent を作る／更新する（課金されない）'
    )
    setup_kb.add_argument(
        '--recreate',
        action='store_true',
        help='預けてある文書を消して登録し直す（corpus/ の本文を直したとき）',
    )
    setup_kb.set_defaults(func=cmd_setup_kb)

    run = sub.add_parser('run', help='質問集を流して計測する')
    run.add_argument(
        'path',
        choices=metrics.known_paths(),
        help='agents = A, custom = B（TTS は --tts で選ぶ）, kb = C。deepgram / openai / google は run custom --tts <社名> の古い呼び方',
    )
    run.add_argument(
        '--tts',
        choices=tts_choices(),
        default='elevenlabs',
        help='B の TTS プロバイダ（既定 elevenlabs）。run custom のときだけ意味を持つ',
    )
    run.add_argument('--scenario', help='質問の id（省略すると全部を順に）')
    run.add_argument(
        '--dry-run', action='store_true', help='接続せず、検索と設定の確認だけ（課金されない）'
    )
    run.add_argument('--no-audio', action='store_true', help='WAV を残さない')
    run.add_argument('--force', action='store_true', help='残クレジットが少なくても実行する')
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        ConfigError,
        credits.CreditsError,
        agent_setup.SetupError,
        agents_path.AgentsPathError,  # kb_path.KbPathError もこれを継いでいる
        custom_path.CustomPathError,  # 各社の *PathError もこれを継いでいる
    ) as exc:
        print(f'エラー: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
