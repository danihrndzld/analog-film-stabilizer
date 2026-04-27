"""Regression tests for the Electron-to-Python CLI contract."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_main_accepts_two_anchor_batch_contract(monkeypatch, capsys):
    import stabilizer_cli

    captured = {}

    def fake_stabilize_folder(**kwargs):
        captured.update(kwargs)
        return {"total_frames": 1}

    monkeypatch.setattr(stabilizer_cli, "stabilize_folder", fake_stabilize_folder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stabilizer_cli.py",
            "--input",
            "in",
            "--output",
            "out",
            "--anchor1-x",
            "1.5",
            "--anchor1-y",
            "2.5",
            "--anchor2-x",
            "3.5",
            "--anchor2-y",
            "4.5",
            "--quality",
            "90",
        ],
    )

    stabilizer_cli.main()

    msg = json.loads(capsys.readouterr().out.strip())
    assert msg["type"] == "done"
    assert captured["anchor1"] == (1.5, 2.5)
    assert captured["anchor2"] == (3.5, 4.5)
    assert captured["jpeg_quality"] == 90


def test_main_rejects_missing_two_anchor_coordinate(monkeypatch, capsys):
    import stabilizer_cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stabilizer_cli.py",
            "--input",
            "in",
            "--output",
            "out",
            "--anchor1-x",
            "1",
            "--anchor1-y",
            "2",
            "--anchor2-x",
            "3",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        stabilizer_cli.main()

    assert exc.value.code == 1
    msg = json.loads(capsys.readouterr().out.strip())
    assert msg["type"] == "error"
    assert "--anchor1-x, --anchor1-y, --anchor2-x and --anchor2-y" in msg["msg"]


def test_main_rejects_legacy_single_anchor_flags(monkeypatch, capsys):
    import stabilizer_cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stabilizer_cli.py",
            "--input",
            "in",
            "--output",
            "out",
            "--anchor-x",
            "1",
            "--anchor-y",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        stabilizer_cli.main()

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr
    assert "--anchor-x" in stderr
