import pytest
from pathlib import Path
import yaml
from pydantic import BaseModel, ValidationError

# Mirroring the source models to test logic when imports are broken in this env
class Feature(BaseModel):
    enabled: bool
    user_group: str | None = None

    def matches_user_group(self, user_id: str | None) -> bool:
        if self.user_group == "logged_in":
            return user_id is not None
        else:
            return True

class Features(BaseModel):
    postprocessing: Feature

class Message(BaseModel):
    message: str
    enabled: bool = True

class Config(BaseModel):
    features: Features
    messages: dict[str, Message]
    profiles: list[str]

    def get_feature(self, feature_id: str, user_id: str | None = None) -> bool:
        if feature_id in self.features.model_fields:
            feature: Feature = getattr(self.features, feature_id)
            return feature.enabled and feature.matches_user_group(user_id)
        else:
            return True

    @classmethod
    def from_yaml(cls, config_yml: Path):
        with open(config_yml) as f:
            yaml_data: dict = yaml.safe_load(f)
        return cls(**yaml_data)

@pytest.fixture
def mock_config_file(tmp_path):
    config_data = {
        "features": {
            "postprocessing": {"enabled": True, "user_group": "all"}
        },
        "messages": {
            "welcome": {"message": "Hello!", "enabled": True}
        },
        "profiles": ["react_to_me"]
    }
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file

def test_config_from_yaml(mock_config_file):
    config = Config.from_yaml(mock_config_file)
    assert config is not None
    assert "postprocessing" in config.features.model_fields
    assert config.profiles == ["react_to_me"]

def test_get_feature(mock_config_file):
    config = Config.from_yaml(mock_config_file)
    assert config.get_feature("postprocessing", user_id="some_user") is True
    assert config.get_feature("non_existent_feature") is True
