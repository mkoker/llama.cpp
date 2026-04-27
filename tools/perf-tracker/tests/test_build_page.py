from pathlib import Path

import build_page


def test_build_page_with_sample_jsonl(tmp_path: Path) -> None:
    input_dir = tmp_path / 'results'
    input_dir.mkdir()

    (input_dir / '2026-04.jsonl').write_text(
        '\n'.join([
            '{"ts":"2026-04-26T01:00:00Z","model_name":"gemma-4-31B","backend":"vulkan","status":"queued"}',
            '{"ts":"2026-04-26T02:00:00Z","model_name":"gemma-4-31B","backend":"vulkan","status":"queued"}',
            '{"ts":"2026-04-26T03:00:00Z","model_name":"qwen3-32B","backend":"hip","status":"queued"}',
            'not-json',
        ]) + '\n'
    )

    out = tmp_path / 'site' / 'index.html'
    build_page.build_page(input_dir, out)

    html = out.read_text()
    assert out.exists()
    assert 'llama.cpp perf tracker' in html
    assert 'Chart' in html
    assert 'gemma-4-31B' in html
    assert 'qwen3-32B' in html
    assert '>2<' in html
    assert '>1<' in html


def test_build_page_handles_empty_input(tmp_path: Path) -> None:
    input_dir = tmp_path / 'results'
    input_dir.mkdir()
    out = tmp_path / 'site' / 'index.html'

    build_page.build_page(input_dir, out)

    html = out.read_text()
    assert 'No benchmark records yet' in html
