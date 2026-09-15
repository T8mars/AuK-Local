from __future__ import annotations

import pytest

from auk_local.duration import estimate_tts_seconds


def test_verified_short_phrase_uses_1_7_seconds():
    assert estimate_tts_seconds("一只小猫在叫啊") == 1.7


def test_duration_estimator_accounts_for_words_and_punctuation():
    assert estimate_tts_seconds("Hello world!") > estimate_tts_seconds("Hello")
    assert estimate_tts_seconds("你好，世界！") > estimate_tts_seconds("你好世界")


def test_duration_estimator_rejects_empty_text_and_clamps_long_text():
    with pytest.raises(ValueError, match="目标文本"):
        estimate_tts_seconds("  ")
    assert estimate_tts_seconds("测" * 1000) == 30.0
