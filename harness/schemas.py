"""Request/response schemas for the YuE2 async task server.

YuE2 rides the facade's existing MUSIC door — task_type ``t2m`` -> engine kind
``music`` -> ``POST /v1/tasks/music/`` -> ``.mp3`` — the same one ACE-Step
serves, so new-api and the facade need no code change. The body therefore
arrives in ACE-Step's shape: new-api flattens the client's ``metadata`` onto the
top level and sets ``prompt``; the facade strips its control keys and injects
``save_result_path``. Unknown keys (``model``, ``user_id``, ACE-Step's
``audio_duration``/``bpm``...) are ignored rather than rejected, because the
facade forwards whatever the caller sent.

Field mapping onto YuE2's ``SongRequest``:
  prompt | style | tags  -> style   (genre, instruments, vocal, language, tempo)
  lyrics                 -> lyrics  (section tags like [Verse] / [Chorus])
  cot                    -> cot     (full | melody | off; default full)
  abc                    -> abc     (a score to render instead of planning one —
                                     the "edit the score, re-render" loop)
  seed / cfg_scale       -> seed / cfg_scale

Cover (task_type ``cover``) arrives through the same door with the facade's
``reference_audio`` input mapped to ``reference_audio_path``. The engine
transcribes that recording into an ABC score with SheetSage2 and renders it with
the requested style and lyrics — upstream's cover workflow, run in-process.

Response field names mirror the LightX2V engine contract because the facade
depends on them verbatim: ``task_id`` / ``task_status`` / ``save_result_path``.
"""
from typing import Optional

from pydantic import BaseModel, Field

# The lyric budget. YuE2 caps the semantic stage at max_tokens=9000 tokens at
# ~25 Hz, i.e. about six minutes of audio, and a lyric that needs more is NOT an
# error upstream: generation stops at the cap and hands back a song that ends
# mid-phrase with only a ``truncated`` flag to show for it. The worker fails
# such a task, but only after minutes of GPU, so the obvious overflows are
# rejected here at submit time instead.
#
# Semantic tokens scale with how much is sung: per character for CJK, per ~2.6
# letters for Latin scripts (first pair of A100 runs: Mandarin 13.5 tokens/char,
# English 5.2 tokens/letter). The budget is counted in these CJK-equivalent
# units so a mixed-language lyric is judged by one number; after weighting,
# English and Mandarin songs of similar length land in the same 19-24
# tokens/unit band.
#
# Measured on A100 over five songs, long lyrics settle at ~11 semantic tokens per
# unit (335 chars -> 4079 tokens / 2:43, 483 chars -> 5746 / 3:50) while short
# ones are stretched to a ~2.5-minute arrangement regardless. 600 units lands
# near 6700 tokens (~4.5 min), a quarter below the cap: headroom for the
# arrangement variance, since the true length is the model's choice.
LATIN_CHAR_WEIGHT = 0.385  # 5.2 / 13.5
MAX_LYRIC_UNITS = 600

# Independent of the audio ceiling: style text goes into the same prompt prefix
# as the lyrics, and a pathological style string would eat the context budget.
MAX_STYLE_CHARS = 2000

# A caller-supplied ABC score (the edit loop) replaces the planning stage, whose
# own budget is 4096 tokens; a score longer than that cannot have come from YuE2.
MAX_ABC_CHARS = 16000

# A cover comes out about as long as its source (3:58 in -> 3:54 out, measured),
# and the semantic stage stops at ~6 minutes. 330 s keeps a cover near 90 % of
# that cap at worst; a longer source would almost surely be cut mid-phrase.
MAX_COVER_SOURCE_SECONDS = 330


def lyric_units(text: str) -> float:
    """CJK-equivalent length of ``text`` for the lyric budget (whitespace free)."""
    cjk = sum(1 for ch in text if "㐀" <= ch <= "鿿" or "가" <= ch <= "힯"
              or "぀" <= ch <= "ヿ")
    other = sum(1 for ch in text if not ch.isspace()) - cjk
    return cjk + other * LATIN_CHAR_WEIGHT


class MusicTaskRequest(BaseModel):
    # Style / genre description. The facade sends `prompt`; YuE2's own names
    # `style` and `tags` are accepted too so a direct caller can use either.
    prompt: Optional[str] = None
    style: Optional[str] = None
    tags: Optional[str] = None

    # Lyrics with section tags. May be empty for an instrumental.
    lyrics: Optional[str] = None

    # Symbolic planning mode: full (melody + chords), melody, or off.
    cot: Optional[str] = None

    # External ABC score to render (requires cot=full|melody).
    abc: Optional[str] = None

    seed: Optional[int] = Field(default=None, ge=0, lt=2**63)
    cfg_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)

    # Output path — the facade always injects this (absolute NFS path, .mp3 for
    # the music kind). The container format follows its extension.
    save_result_path: Optional[str] = None

    # Cover source recording (absolute NFS path materialised by new-api). The
    # facade strips task_type as a control key, so the presence of this path is
    # the only sign that a cover — not a fresh song — was asked for.
    reference_audio_path: Optional[str] = None

    # Declared only to be refused: ACE-Step's repaint (src_audio ->
    # src_audio_path) reaches the same door. Ignored like other unknown keys,
    # it would come back as an unrelated fresh song marked successful.
    src_audio_path: Optional[str] = None

    # Engine-native id; the facade lets the engine generate it.
    task_id: Optional[str] = None

    def style_text(self) -> str:
        return (self.prompt or self.style or self.tags or "").strip()

    def lyrics_text(self) -> str:
        return (self.lyrics or "").strip()

    def is_cover(self) -> bool:
        return bool(self.reference_audio_path)

    def cot_mode(self) -> str:
        # Covers default to melody-only, upstream's recommendation for a style
        # change: the transcribed harmony stays out of the way of the new
        # accompaniment. cot=full keeps the source's chords as well.
        default = "melody" if self.is_cover() else "full"
        return (self.cot or default).strip().lower()

    def abc_text(self) -> Optional[str]:
        return self.abc if self.abc and self.abc.strip() else None


class TaskResponse(BaseModel):
    task_id: str
    task_status: str = "pending"
    save_result_path: Optional[str] = None


class StopTaskResponse(BaseModel):
    stop_status: str  # "success" | "do_nothing" | "error"
    reason: str = ""
