import pytest
from bl_ext.user_default.UVgami.src.engines import binary_engine
from bl_ext.user_default.UVgami.src.engines.binary_engine import (
    BinaryEngine,
    EngineRelease,
    parse_version,
)
from bl_ext.user_default.UVgami.src.utils.paths import get_engine_binary_name

VERSION = "1.20.2"
MINIMUM_VERSION = "1.20.0"


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(binary_engine, "get_extension_dir_path", lambda: tmp_path)
    return EngineRelease("optcuts", "Optcuts", VERSION, MINIMUM_VERSION, "2 MB")


@pytest.fixture
def install(tmp_path):
    def make(version):
        directory = tmp_path / "optcuts" / version
        directory.mkdir(parents=True)
        (directory / get_engine_binary_name("optcuts")).touch()

    return make


def test_parse_version_rejects_a_name_that_is_not_a_version():
    assert parse_version("1.20.2") == (1, 20, 2)
    assert parse_version("backup") is None
    assert parse_version("1.20") is None


def test_version_order_is_numeric():
    assert parse_version("1.9.0") < parse_version("1.20.0")


def test_nothing_downloaded(release):
    assert release.installed_version() is None
    assert release.installed_path() is None
    assert not release.install_too_old()
    assert not release.update_available()


def test_pinned_version_runs(release, install):
    install(VERSION)
    assert release.installed_path().is_file()
    assert not release.update_available()


def test_patch_behind_still_runs(release, install):
    install("1.20.1")
    assert release.installed_path().is_file()
    assert release.update_available()
    assert not release.install_too_old()


def test_newer_than_pinned_runs_without_a_notice(release, install):
    install("1.21.0")
    assert release.installed_path().is_file()
    assert not release.update_available()
    assert not release.install_too_old()


def test_below_minimum_refuses_to_run(release, install):
    install("1.19.9")
    assert release.installed_path() is None
    assert release.install_too_old()


def test_missing_binary_is_not_an_install(release, tmp_path):
    (tmp_path / "optcuts" / VERSION).mkdir(parents=True)
    assert release.installed_path() is None
    assert not release.install_too_old()


def test_a_dead_download_falls_back_to_the_working_install(release, install, tmp_path):
    install("1.20.1")
    # killed mid download: the folder is made before the binary lands in it
    (tmp_path / "optcuts" / VERSION).mkdir()
    assert release.installed_version() == "1.20.1"
    assert release.installed_path().is_file()
    assert release.update_available()


def test_a_folder_that_is_not_a_version_is_ignored(release, tmp_path):
    (tmp_path / "optcuts" / "backup").mkdir(parents=True)
    assert release.installed_version() is None


def test_local_build_never_asks_for_an_update(release, install, monkeypatch):
    install("1.20.1")
    engine = BinaryEngine()
    engine.release = release
    monkeypatch.setattr(binary_engine, "get_local_engine_path", lambda name: None)
    assert engine.update_pending()
    # this checkout has its own build
    monkeypatch.setattr(binary_engine, "get_local_engine_path", lambda name: "engine")
    assert not engine.update_pending()
