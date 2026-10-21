"""Resolve and play a voice's sample audio for a given TTS model (and language).

Resolution is exactly :meth:`VoxFile.resolve_sample_audio` (Swift
``sampleAudioData(for:language:)``): the query is a case-insensitive substring of the
embedding key or model (``0.6b``, ``1.7b``, ``qwen3-tts-1.7b``, ``gemini``,
``gemini-3.8-flash-tts``), and the language follows exact -> base -> default -> legacy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ._query import Resolved
from .errors import PlayerNotFoundError, SampleNotFoundError
from .voxfile import VoxFile

#: Player candidates in preference order: (binary, argv prefix).
PLAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("afplay", ()),
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "error")),
    ("aplay", ("-q",)),
)


@dataclass(frozen=True)
class PlayableSample:
    key: str
    model: str
    engine: str | None
    language: str | None  # None = default / language-neutral
    file: str
    present: bool
    size: int


def playable_samples(vox: VoxFile) -> list[PlayableSample]:
    """Every sample-audio embedding declared in the manifest (plus a legacy sample)."""
    out = []
    for e in vox.embedding_entries:
        if "sample-audio" not in e.file:
            continue
        entry = vox[e.file]
        out.append(PlayableSample(e.key, e.model, e.engine, e.language, e.file, entry is not None,
                                  entry.size if entry else 0))
    from ._query import LEGACY_SAMPLE_AUDIO

    if not out and vox[LEGACY_SAMPLE_AUDIO] is not None:
        entry = vox[LEGACY_SAMPLE_AUDIO]
        out.append(PlayableSample("(legacy)", "qwen3-tts", "qwen3-tts", None, LEGACY_SAMPLE_AUDIO, True, entry.size))
    return out


def resolve(vox: VoxFile, query: str, language: str | None = None) -> Resolved:
    """Resolve the sample or raise :class:`SampleNotFoundError` with a helpful message."""
    r = vox.resolve_sample_audio(query, language)
    if r is None:
        available = sorted({f"{s.key}" for s in playable_samples(vox)})
        hint = f" Available samples: {', '.join(available)}" if available else " This voice has no sample audio."
        lang = f" (language {language})" if language else ""
        raise SampleNotFoundError(f"No sample audio for model '{query}'{lang}.{hint}")
    if r.data is None:
        raise SampleNotFoundError(f"Manifest declares '{r.path}' for '{query}' but the file is missing from the archive")
    return r

def find_player(preferred: str | None = None, which: Callable[[str], str | None] | None = None) -> list[str]:
    """argv prefix for the first available player (or ``preferred``)."""
    which = which or shutil.which
    if preferred:
        exe = which(preferred)
        if not exe:
            raise PlayerNotFoundError(f"audio player '{preferred}' not found on PATH")
        args = next((a for n, a in PLAYERS if n == Path(preferred).name), ())
        return [exe, *args]
    for name, args in PLAYERS:
        exe = which(name)
        if exe:
            return [exe, *args]
    raise PlayerNotFoundError(
        "no audio player found (tried afplay, ffplay, aplay); use --extract PATH to write the WAV instead"
    )


def extract(vox: VoxFile, query: str, dest: str | os.PathLike[str], language: str | None = None) -> tuple[Resolved, Path]:
    r = resolve(vox, query, language)
    out = Path(dest)
    if out.is_dir():
        out = out / Path(r.path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.data or b"")
    return r, out


def play(vox: VoxFile, query: str, language: str | None = None, *, player: str | None = None,
         runner: Callable[..., Any] | None = None, which: Callable[[str], str | None] | None = None) -> Resolved:
    """Play the resolved sample with the system player (blocks until playback ends)."""
    r = resolve(vox, query, language)
    argv = find_player(player, which)
    suffix = Path(r.path).suffix or ".wav"
    fd, tmp = tempfile.mkstemp(prefix="voxpy-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(r.data or b"")
        (runner or subprocess.run)([*argv, tmp], check=True)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return r


def play_file(path: str | os.PathLike[str], query: str, language: str | None = None, **kwargs: Any) -> Resolved:
    return play(VoxFile.open(path), query, language, **kwargs)


__all__: Sequence[str] = ("PlayableSample", "playable_samples", "resolve", "find_player", "extract", "play", "play_file")
