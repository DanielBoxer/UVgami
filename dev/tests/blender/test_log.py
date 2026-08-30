"""The log file under the extension dir, which the Log button renders. One
block per .blend, so a run can be traced to the file it was on."""

import bpy
import pytest
from bl_ext.user_default.UVgami.src import logger as log_module
from bl_ext.user_default.UVgami.src.logger import (
    LOG_LINE_LIMIT,
    UNSAVED_FILE_NAME,
    block_header,
    split_blocks,
    trim_blocks,
)
from bl_ext.user_default.UVgami.src.ops.info import LOG_TEXT_NAME
from blender_fixtures import logger, needs_engine

OLD_BLOCK = (
    "UVgami 0.1 | Blender 1 | Linux | 2020-01-01 00:00:00 | old.blend"
    "\n00:00:01 | old run\n"
)


@pytest.fixture(autouse=True)
def log_file(empty_scene, tmp_path, monkeypatch):
    """Keep the tests off the log the installed addon writes."""
    path = tmp_path / "log.txt"
    monkeypatch.setattr(log_module, "get_log_path", lambda: path)
    logger.reset()
    return path


def log_text():
    return bpy.data.texts[LOG_TEXT_NAME].as_string()


def headers(content):
    return [block[0] for block in split_blocks(content)]


def test_split_blocks_keeps_each_block_with_its_header():
    content = "UVgami 1 | a\n12:00:00 | one\n\nUVgami 2 | b\n12:00:01 | two\n"
    assert split_blocks(content) == [
        ["UVgami 1 | a", "12:00:00 | one"],
        ["UVgami 2 | b", "12:00:01 | two"],
    ]


def test_trim_blocks_drops_the_oldest_past_the_limit():
    blocks = [["header"] + ["line"] * 300 for _ in range(3)]
    assert trim_blocks(blocks) == blocks[2:]


def test_trim_blocks_keeps_the_newest_whole():
    newest = ["header"] + ["line"] * (LOG_LINE_LIMIT * 2)
    assert trim_blocks([["UVgami old"], newest]) == [newest]


def test_header_names_the_blend():
    assert block_header().endswith(f"| {UNSAVED_FILE_NAME}")


def test_finished_run_goes_under_a_header(log_file):
    logger.new_info()
    logger.write_latest()

    assert split_blocks(log_file.read_text()) == [
        [logger.header, logger.get_latest().get_info()]
    ]


def test_second_run_on_the_same_file_joins_the_block(log_file):
    logger.new_info()
    logger.write_latest()
    logger.new_info()
    logger.write_latest()

    assert len(split_blocks(log_file.read_text())) == 1


def test_a_file_load_starts_a_block_naming_the_new_file(log_file, tmp_path):
    logger.new_info()
    logger.write_latest()
    first = logger.header

    saved = tmp_path / "second.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(saved))
    bpy.ops.wm.open_mainfile(filepath=str(saved))
    logger.new_info()
    logger.write_latest()

    assert headers(log_file.read_text()) == [first, logger.header]
    assert logger.header.endswith("| second.blend")


def test_an_earlier_block_stays_above_this_one(log_file):
    log_file.write_text(OLD_BLOCK)
    logger.new_info()

    logger.write_latest()

    blocks = split_blocks(log_file.read_text())
    assert blocks[0] == OLD_BLOCK.splitlines()
    assert blocks[1][0] == logger.header


def test_log_says_so_when_nothing_ran():
    bpy.ops.uvgami.open_logs()
    assert log_text().strip() == "No previous unwraps"


def test_log_shows_a_run_the_file_has_not_got_yet():
    logger.new_info()

    bpy.ops.uvgami.open_logs()

    assert "In Progress" in log_text()


def test_log_survives_a_file_load():
    logger.new_info()
    logger.write_latest()
    line = logger.get_latest().get_info()

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.uvgami.open_logs()

    assert line in log_text()


@needs_engine
@pytest.mark.smoke
def test_log_line_names_the_input(log_file, unwrap):
    bpy.ops.mesh.primitive_cube_add()
    bpy.context.active_object.name = "cube"

    unwrap()

    line = log_file.read_text().splitlines()[-1]
    assert "cube (12 tris)" in line
    assert "Pieces: 1 finished" in line
