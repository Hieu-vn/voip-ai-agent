import struct

import pytest

from src.core.call_handler import EnergyVAD


def make_chunk(amplitude: int, length: int = 160) -> bytes:
    return b"".join(struct.pack("<h", amplitude) for _ in range(length))


def test_energy_vad_triggers_on_loud_audio():
    vad = EnergyVAD(energy_threshold=500, activation_frames=2, release_frames=3)
    silence = make_chunk(10)
    loud = make_chunk(2000)

    assert not vad.add_chunk(silence)
    assert not vad.add_chunk(silence)

    assert not vad.add_chunk(loud)
    assert vad.add_chunk(loud)


def test_energy_vad_resets_after_silence():
    vad = EnergyVAD(energy_threshold=500, activation_frames=1, release_frames=2)
    loud = make_chunk(1500)
    silence = make_chunk(0)

    assert vad.add_chunk(loud)
    assert not vad.add_chunk(silence)
    assert not vad.add_chunk(silence)
    assert not vad.add_chunk(silence)
    assert vad.add_chunk(loud)
