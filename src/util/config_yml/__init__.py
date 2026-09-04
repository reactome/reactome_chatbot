from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ValidationError

from agent.profile_names import ProfileName
from util.config_yml.features import Feature, Features
from util.config_yml.messages import Message, TriggerEvent
from util.config_yml.usage_limits import MessageRate, UsageLimits
from util.config_yml.user_matching import match_user
from util.logging import logging

# Anchored to the repo rather than the working directory, so these resolve the
# same whether the process starts from the repo root, a subdirectory, or /app in
# the container. Matches util.embedding_environment.REPO_ROOT.
REPO_ROOT: Path = Path(__file__).parent.parent.parent.parent
CONFIG_YML = REPO_ROOT / "config.yml"
CONFIG_DEFAULT_YML = REPO_ROOT / "config_default.yml"


class Config(BaseModel):
    features: Features
    messages: dict[str, Message]
    profiles: list[ProfileName]
    usage_limits: UsageLimits

    def get_feature(
        self,
        feature_id: str,
        user_id: str | None = None,
    ) -> bool:
        if feature_id in self.features.model_fields:
            feature: Feature = getattr(self.features, feature_id)
            return feature.enabled and feature.matches_user_group(user_id)
        return True

    def get_messages(
        self,
        user_id: str | None = None,
        event: TriggerEvent | None = None,
        after_messages: int | None = None,
        last_messages: dict[str, str] | None = None,
    ) -> dict[str, str]:
        last_messages = last_messages if last_messages is not None else {}
        return {
            message_id: message.message
            for message_id, message in self.messages.items()
            if (
                message.enabled
                and match_user(message.recipients, user_id)
                and message.trigger.match_trigger(
                    event, after_messages, last_messages.get(message_id)
                )
            )
        }

    def get_message_rate_usage_limited(
        self,
        user_id: str | None = None,
        message_times_queue: list[str] | None = None,
    ) -> MessageRate | None:
        message_times_queue = (
            message_times_queue if message_times_queue is not None else []
        )
        message_rate: MessageRate
        for message_rate in self.usage_limits.message_rates:
            if match_user(message_rate.users, user_id):
                return message_rate.check_rate(message_times_queue)
        return None  # not rate limited

    @classmethod
    def _load(cls, config_yml: Path) -> Self:
        with open(config_yml) as f:
            yaml_data = yaml.safe_load(f)
        if not isinstance(yaml_data, dict):
            raise ValueError(f"{config_yml} is empty or is not a YAML mapping")
        return cls(**yaml_data)

    @classmethod
    def from_yaml(cls, config_yml: Path = CONFIG_YML) -> Self | None:
        """Load config.yml, falling back to the shipped defaults.

        A None return disables every config-driven feature *including rate
        limiting*, so a broken config.yml must not land there if the defaults are
        usable -- an unreadable file should not quietly remove the message quota.
        Note config.yml is a bind mount in docker-compose: if the file is missing
        on the host, Docker creates a directory in its place, which is why
        IsADirectoryError is handled alongside a genuine absence.
        """
        if config_yml != CONFIG_DEFAULT_YML:
            try:
                return cls._load(config_yml)
            except FileNotFoundError:
                logging.warning(
                    f"Config file not found: {config_yml} ; "
                    f"falling back to {CONFIG_DEFAULT_YML}"
                )
            except (
                ValidationError,
                ValueError,
                IsADirectoryError,
                yaml.YAMLError,
            ) as e:
                logging.error(
                    f"Invalid config {config_yml}; falling back to "
                    f"{CONFIG_DEFAULT_YML}. Fix this -- the fallback is not what "
                    f"you configured:\n{e}"
                )

        try:
            return cls._load(CONFIG_DEFAULT_YML)
        except Exception as e:
            logging.error(
                f"Default config {CONFIG_DEFAULT_YML} is unusable, so all "
                f"config-driven features including rate limiting are OFF:\n{e}"
            )
            return None
