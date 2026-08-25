"""Where a project's taxonomy.json lives."""

from __future__ import annotations

from pathlib import Path

from littax.config import PROJECT_DIR, Config, has_project


class TestOutDir:
    def test_derives_from_the_source_folder(self):
        cfg = Config(source_root=Path("papers"))
        assert cfg.out == Path("papers") / PROJECT_DIR

    def test_follows_the_source_folder(self):
        """The UI analyses a different folder per run by setting source_root."""
        cfg = Config()
        cfg.source_root = Path("elsewhere")
        assert cfg.out == Path("elsewhere") / PROJECT_DIR

    def test_an_explicit_out_dir_wins(self):
        cfg = Config(source_root=Path("papers"), out_dir=Path("somewhere/else"))
        assert cfg.out == Path("somewhere/else")

    def test_a_toml_out_dir_still_works(self, tmp_path):
        """Configs written before the move must keep behaving."""
        toml = tmp_path / "config.toml"
        toml.write_text('source_root = "raw"\nout_dir = "out"\n', encoding="utf-8")
        assert Config.load(toml).out == Path("out")


class TestHasProject:
    def test_true_only_with_a_taxonomy(self, tmp_path):
        assert has_project(tmp_path) is False

        (tmp_path / PROJECT_DIR).mkdir()
        assert has_project(tmp_path) is False  # an empty dir is not a project

        (tmp_path / PROJECT_DIR / "taxonomy.json").write_text("{}", encoding="utf-8")
        assert has_project(tmp_path) is True

    def test_false_for_a_missing_folder(self, tmp_path):
        assert has_project(tmp_path / "nope") is False
