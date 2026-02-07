"""Tests for the unified CLI entry point (skseq)."""

from typer.testing import CliRunner

from skewed_sequences.cli import app

runner = CliRunner()


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "skseq" in result.output.lower() or "skewed" in result.output.lower()


def test_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.output.lower()


def test_visualize_help():
    result = runner.invoke(app, ["visualize", "--help"])
    assert result.exit_code == 0


def test_data_help():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0


def test_experiments_help():
    result = runner.invoke(app, ["experiments", "--help"])
    assert result.exit_code == 0
