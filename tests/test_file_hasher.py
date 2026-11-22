import pytest

from app.ingestion.file_hasher import calculate_file_hash


def test_same_content_produces_same_hash(tmp_path):
    first_file = tmp_path / "first.pdf"
    second_file = tmp_path / "renamed.pdf"

    first_file.write_bytes(b"same PDF content")
    second_file.write_bytes(b"same PDF content")

    assert calculate_file_hash(first_file) == calculate_file_hash(second_file)


def test_different_content_produces_different_hash(tmp_path):
    first_file = tmp_path / "first.pdf"
    second_file = tmp_path / "second.pdf"

    first_file.write_bytes(b"PDF content A")
    second_file.write_bytes(b"PDF content B")

    assert calculate_file_hash(first_file) != calculate_file_hash(second_file)


def test_missing_file_raises_error(tmp_path):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        calculate_file_hash(missing_file)


def test_directory_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        calculate_file_hash(tmp_path)
