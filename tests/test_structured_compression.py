import random

from app.structured_compression.compressor import StructuredCompressor

compressor = StructuredCompressor()


def roundtrip(text: str) -> None:
    assert compressor.decompress(compressor.compress(text)) == text


def test_empty_and_plain_text_survive():
    for text in ["", "plain text only", "two\nlines", "\n\n"]:
        roundtrip(text)


def test_awkward_characters_fall_back_to_raw_lines():
    roundtrip("x=1|2 z\nweird {} line 5\n!starts with a bang\nk=v")


def test_repeated_lines_share_one_template():
    text = "\n".join(f"10:0{i} INFO user={i} status=200" for i in range(5))
    packed = compressor.compress(text)
    assert packed.count("@") == 1
    assert "status=200" in packed.split("---")[0]
    roundtrip(text)


def test_random_logs_round_trip():
    rng = random.Random(3)
    for _ in range(200):
        lines = [
            f"2026-11-12 10:{rng.randint(0, 59):02d}:00 {rng.choice(['INFO', 'WARN'])} id={rng.randint(1, 10**8)} "
            f"path=/a/{rng.choice(['x', 'y'])}/{rng.randint(1, 9)}"
            for _ in range(rng.randint(1, 12))
        ]
        roundtrip("\n".join(lines))
