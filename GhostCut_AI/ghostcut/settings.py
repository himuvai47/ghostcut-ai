from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from .utils import read_json, write_json

@dataclass(slots=True)
class GhostCutSettings:
    app_root: str
    facelessrc_root: str
    agent1_root: str
    agent2_root: str
    agent3_root: str
    projects_root: str
    indexes_root: str
    ollama_url: str = "http://127.0.0.1:11434"
    auto_open_browser: bool = True
    host: str = "127.0.0.1"
    port: int = 8765

    @classmethod
    def defaults(cls, app_root: Path) -> "GhostCutSettings":
        # GhostCut_AI is expected next to Agent1/2/3 under D:\\AI\\FacelessRC.
        faceless = app_root.parent
        return cls(
            app_root=str(app_root.resolve()),
            facelessrc_root=str(faceless.resolve()),
            agent1_root=str((faceless / "Agent1_Indexing").resolve()),
            agent2_root=str((faceless / "Agent2_VideoCreation").resolve()),
            agent3_root=str((faceless / "Agent3_Renderer").resolve()),
            projects_root=str((faceless / "GhostCutProjects").resolve()),
            indexes_root=str((faceless / "indexes").resolve()),
        )

class SettingsStore:
    def __init__(self, app_root: Path):
        self.app_root = app_root
        self.path = app_root / "data" / "settings.json"
        self._settings = self._load()

    def _load(self) -> GhostCutSettings:
        base = GhostCutSettings.defaults(self.app_root)
        data = read_json(self.path, {}) or {}
        for k, v in data.items():
            if hasattr(base, k): setattr(base, k, v)
        self.save(base)
        return base

    @property
    def value(self) -> GhostCutSettings:
        return self._settings

    def save(self, settings: GhostCutSettings | None = None) -> None:
        if settings is not None: self._settings = settings
        write_json(self.path, asdict(self._settings))

    def update(self, data: dict) -> GhostCutSettings:
        for k, v in data.items():
            if hasattr(self._settings, k) and k not in {"app_root"}:
                setattr(self._settings, k, v)
        self.save()
        return self._settings
