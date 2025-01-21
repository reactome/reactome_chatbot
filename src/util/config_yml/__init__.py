from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ValidationError

from util.config_yml.features import Feature, Features
from util.config_yml.messages import Message, TriggerEvent
from util.logging import logging

CONFIG_YML = Path("config.yml")
CONFIG_DEFAULT_YML = Path("config_default.yml")


class Config(BaseModel):
    features: Features
    messages: dict[str, Message]

    def get_feature(
        self,
        feature_id: str,
        user_id: str | None = None,
    ) -> bool:
        if feature_id in self.features.model_fields:
            feature: Feature = getattr(self.features, feature_id)
            return feature.enabled and feature.matches_user_group(user_id)
        else:
            return True

    def get_messages(
        self,
        user_id: str | None = None,
        event: TriggerEvent | None = None,
        after_messages: int | None = None,
        last_messages: dict[str, str] = {},
    ) -> dict[str, str]:
        return {
            message_id: message.message
            for message_id, message in self.messages.items()
            if (
                message.enabled
                and message.match_recipient(user_id)
                and message.trigger.match_trigger(
                    event, after_messages, last_messages.get(message_id, None)
                )
            )
        }

    @classmethod
    def from_yaml(cls, config_yml: Path = CONFIG_YML) -> Self | None:
        if not config_yml.exists():
            logging.warning(
                f"Config file not found: {config_yml} ; falling back to {CONFIG_DEFAULT_YML}"
            )
            config_yml = CONFIG_DEFAULT_YML
        with open(config_yml) as f:
            yaml_data: dict = yaml.safe_load(f)
        try:
            return cls(**yaml_data)
        except ValidationError as e:
            logging.warning(e)
            return None
