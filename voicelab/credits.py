"""クレジット残高の読み取り。

会話は分単位でクレジットを消費するので、**実行の前後で残高を読んで差を残す**。
読み取り自体（`GET /v1/user/subscription`）は課金されない。

会話ごとの正確な費用は、会話の詳細（`GET /v1/convai/conversations/{id}`）の
``metadata.cost`` にある。残高の反映は遅れるので、費用はそちらを正とする。

HTTP は ``httpx``。最初は「骨組みは標準ライブラリだけで動かす」方針で ``urllib`` を
使っていたが、`elevenlabs` / `google-genai` が必須依存になった時点でその理由は消えた。
repo 内の HTTP を 1 つに揃えるため、ほかのモジュールと同じ ``httpx`` にしている
（``requests`` も間接的に入ってはいるが、宣言していないものは import しない）。
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

API_BASE = 'https://api.elevenlabs.io/v1'
TIMEOUT_SECONDS = 30


class CreditsError(RuntimeError):
    """残高を読めなかった（キー不正、接続失敗、応答の形が違う）。"""


@dataclass(frozen=True)
class CreditsSnapshot:
    """ある時点の残高。

    :ivar tier: プラン名（``free`` / ``starter`` …）。
    :ivar used: 今期に使ったクレジット。
    :ivar limit: 今期の上限。
    :ivar resets_at: 次のリセット日時（UTC）。
    :ivar taken_at: 読んだ日時（UTC）。
    """

    tier: str
    used: int
    limit: int
    resets_at: datetime
    taken_at: datetime

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    def describe(self) -> str:
        """人が読む 1 行。"""
        return (
            f'{self.tier}: {self.used:,} / {self.limit:,} 使用（残り {self.remaining:,}）'
            f' リセット {self.resets_at:%Y-%m-%d %H:%M} UTC'
        )


def read_subscription(api_key: str) -> CreditsSnapshot:
    """残高を 1 回読む。

    :raises CreditsError: 2xx 以外、接続失敗、または必要な項目が無い。
    """
    try:
        response = httpx.get(
            f'{API_BASE}/user/subscription',
            headers={'xi-api-key': api_key, 'accept': 'application/json'},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise CreditsError(
            f'ElevenLabs が HTTP {exc.response.status_code} を返しました'
        ) from exc
    except httpx.HTTPError as exc:
        raise CreditsError('ElevenLabs に接続できませんでした') from exc
    except ValueError as exc:  # JSON として読めない本文
        raise CreditsError('残高の応答が JSON ではありません') from exc

    try:
        return CreditsSnapshot(
            tier=str(payload['tier']),
            used=int(payload['character_count']),
            limit=int(payload['character_limit']),
            resets_at=datetime.fromtimestamp(
                int(payload['next_character_count_reset_unix']), tz=timezone.utc
            ),
            taken_at=datetime.now(timezone.utc),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CreditsError('残高の応答に必要な項目がありません') from exc


def consumed(before: CreditsSnapshot, after: CreditsSnapshot) -> int:
    """2 回の読み取りの差（消費クレジット）。期をまたいでリセットされた場合は after の使用量。"""
    if after.resets_at != before.resets_at:
        return after.used
    return max(after.used - before.used, 0)


# --------------------------------------------------------------------------- Deepgram（D）

DEEPGRAM_API_BASE = 'https://api.deepgram.com/v1'


@dataclass(frozen=True)
class DeepgramBalance:
    """Deepgram の残高（ドル）。ElevenLabs のクレジットとは別勘定。

    :ivar amount: 全 balance の合計。``balances`` が空配列なら 0（**残高なし**）。
    :ivar units: 単位（``usd``）。空配列のときは ``usd`` とみなす。
    :ivar entries: balance の件数。0 なら「残高なし」と表示する。
    """

    amount: float
    units: str
    entries: int

    def describe(self) -> str:
        """人が読む 1 行。"""
        if self.entries == 0:
            return 'Deepgram: 残高なし（balances が空。TTS は 402 で断られる）'
        return f'Deepgram: 残り {self.amount:,.4f} {self.units}'


def parse_deepgram_balances(payload: dict) -> DeepgramBalance:
    """``GET /v1/projects/{id}/balances`` の応答を読む純粋関数。

    応答は ``{"balances": [{"amount": 199.9, "units": "usd", ...}]}``。
    **空配列は異常ではなく「残高なし」**（無料枠が失効するとこうなる。2026-09-25 に実測）。

    :raises CreditsError: 必要な項目が無い。
    """
    try:
        balances = payload['balances']
        amount = sum(float(item['amount']) for item in balances)
        units = str(balances[0].get('units', 'usd')) if balances else 'usd'
    except (KeyError, TypeError, ValueError) as exc:
        raise CreditsError('Deepgram の残高の応答に必要な項目がありません') from exc
    return DeepgramBalance(amount=amount, units=units, entries=len(balances))


def read_deepgram_balance(api_key: str) -> DeepgramBalance:
    """Deepgram の残高を 1 回読む（課金なし）。先頭のプロジェクトを見る。

    鍵に ``billing:read`` の権限が無いと ``/balances`` は **403** を返す。呼び出し側
    （``cli``）はこの :class:`CreditsError` を 1 行の注意として出して先へ進む。
    D の計測は残高を止まる条件にしていないので、読めなくても困らない。

    :raises CreditsError: 2xx 以外、接続失敗、または応答の形が違う。
    """
    headers = {'Authorization': f'Token {api_key}', 'accept': 'application/json'}
    try:
        with httpx.Client(headers=headers, timeout=TIMEOUT_SECONDS) as client:
            projects = client.get(f'{DEEPGRAM_API_BASE}/projects')
            projects.raise_for_status()
            items = projects.json().get('projects') or []
            if not items:
                raise CreditsError('Deepgram のプロジェクトが見つかりません')
            project_id = items[0]['project_id']
            response = client.get(f'{DEEPGRAM_API_BASE}/projects/{project_id}/balances')
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        hint = '。鍵に billing:read の権限が無い' if status == 403 else ''
        raise CreditsError(f'Deepgram が HTTP {status} を返しました{hint}') from exc
    except httpx.HTTPError as exc:
        raise CreditsError('Deepgram に接続できませんでした') from exc
    except (KeyError, ValueError) as exc:
        raise CreditsError('Deepgram の応答が想定の形ではありません') from exc
    return parse_deepgram_balances(payload)


if __name__ == '__main__':
    # python -m voicelab.credits
    # config をここで import するのは、この module 自体を .env に依存させないため
    # （ライブラリとしては api_key を引数で受け取るだけにしておきたい）。
    from .config import require

    print(read_subscription(require('ELEVENLABS_API_KEY')).describe())
