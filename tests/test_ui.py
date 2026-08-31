"""Smoke tests for the Streamlit app.

The failure these exist to catch is the app rendering nothing but loading
skeletons — a hang in the script thread, which no other test would notice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="pip install 'pathora-ai[ui]'")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src/pathora/ui/app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    return at


def test_app_renders_without_exception(app):
    assert not app.exception


def test_title_and_tagline_present(app):
    assert "Pathora AI" in [t.value for t in app.title]
    assert any("Understand your profile" in c.value for c in app.caption)


def test_build_stamp_is_shown(app):
    """Makes a stale server visibly stale instead of mysteriously stale."""
    assert any("ui file" in c.value for c in app.caption)


def test_sidebar_inputs_render(app):
    assert len(app.sidebar) > 5


def test_script_completes_rather_than_hanging(app):
    """A cached event loop across Streamlit's threads used to block here."""
    assert app.get("file_uploader")


class TestDeployability:
    """Streamlit Community Cloud installs requirements.txt and runs streamlit_app.py."""

    def test_entrypoint_exists_and_renders(self):
        root = Path(__file__).resolve().parents[1]
        at = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=90)
        at.run()
        assert not at.exception
        assert "Pathora AI" in [t.value for t in at.title]

    def test_requirements_cover_every_runtime_import(self):
        """A missing pin here fails at deploy time, not in local dev."""
        root = Path(__file__).resolve().parents[1]
        requirements = "\n".join(
            line.split("#")[0]
            for line in (root / "requirements.txt").read_text().lower().splitlines()
        )
        for package in (
            "pydantic",
            "pydantic-settings",
            "langgraph",
            "langchain-core",
            "pymupdf",
            "httpx",
            "streamlit",
        ):
            assert package in requirements, f"{package} missing from requirements.txt"

    def test_requirements_exclude_heavy_optional_infra(self):
        """The deployed demo must not need Postgres, Redis or Pinecone."""
        root = Path(__file__).resolve().parents[1]
        pinned = [
            line.split("#")[0].strip().lower()
            for line in (root / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for package in ("psycopg", "redis", "pinecone", "sqlalchemy"):
            assert not any(package in line for line in pinned), f"{package} should be optional"


class TestWhatIfLab:
    """Failure modes that all looked like 'the button does nothing'."""

    def test_major_field_is_constrained_not_free_text(self):
        """Free text accepted 'Dara Science' and ran on a nonexistent discipline."""
        source = (Path(__file__).resolve().parents[1] / "src/pathora/ui/app.py").read_text()
        assert 'st.selectbox(\n            "Simulated intended major"' in source
        assert '"Simulated intended major", ""' not in source

    def test_errors_are_surfaced_not_swallowed(self):
        source = (Path(__file__).resolve().parents[1] / "src/pathora/ui/app.py").read_text()
        assert "Simulation failed:" in source
        assert "Re-run the analysis and try again" in source

    def test_results_persist_across_reruns(self):
        """Rendering inside the button block wiped results on the next rerun."""
        source = (Path(__file__).resolve().parents[1] / "src/pathora/ui/app.py").read_text()
        assert 'st.session_state["whatif"] = result' in source
        assert 'st.session_state.get("whatif")' in source

    def test_spinner_shown_during_the_run(self):
        source = (Path(__file__).resolve().parents[1] / "src/pathora/ui/app.py").read_text()
        assert "st.spinner(" in source


class TestHuggingFaceSpace:
    """Spaces needs frontmatter and the right port, or the build fails opaquely."""

    def _space(self, name: str) -> str:
        return (Path(__file__).resolve().parents[1] / "deploy/huggingface" / name).read_text()

    def test_readme_starts_with_frontmatter(self):
        assert self._space("README.md").startswith("---\n")

    def test_frontmatter_declares_docker_sdk_and_port(self):
        front = self._space("README.md").split("---")[1]
        assert "sdk: docker" in front
        assert "app_port: 7860" in front

    def test_dockerfile_listens_on_the_spaces_port(self):
        dockerfile = self._space("Dockerfile")
        assert "STREAMLIT_SERVER_PORT=7860" in dockerfile
        assert "EXPOSE 7860" in dockerfile

    def test_dockerfile_runs_as_a_non_root_user(self):
        """Spaces only grants write access under the user's home."""
        dockerfile = self._space("Dockerfile")
        assert "useradd" in dockerfile
        assert "USER user" in dockerfile

    def test_dockerfile_runs_streamlit_not_the_api(self):
        assert "streamlit" in self._space("Dockerfile").split("CMD")[-1]
