from __future__ import annotations

from pnghelpers import make_png, make_png_ztxt

from argus_proof.backends.pnginfo import read_dimensions, read_text_chunks


def test_reads_text_chunks() -> None:
    png = make_png(text={"prompt": '{"3": {"seed": 42}}', "workflow": "ui-graph"})
    chunks = read_text_chunks(png)
    assert chunks["prompt"] == '{"3": {"seed": 42}}'
    assert chunks["workflow"] == "ui-graph"


def test_reads_compressed_ztxt() -> None:
    assert read_text_chunks(make_png_ztxt("prompt", "compressed value"))["prompt"] == "compressed value"


def test_reads_dimensions_from_ihdr() -> None:
    assert read_dimensions(make_png(width=1024, height=768)) == (1024, 768)


def test_non_png_returns_empty_and_none() -> None:
    assert read_text_chunks(b"not a png") == {}
    assert read_dimensions(b"not a png") is None


def test_no_text_chunks_returns_empty() -> None:
    assert read_text_chunks(make_png()) == {}
