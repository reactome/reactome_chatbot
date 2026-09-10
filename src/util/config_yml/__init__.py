from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from agent.profile_names import ProfileName
from util.config_yml.features import Feature, Features
from util.config_yml.messages import Message, TriggerEvent
from util.config_yml.models import LLMConfig
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
    # extra="forbid" for the same reason LLMConfig does it, one level up. Without
    # it a typo in a section name -- `llmm:` for `llm:`, or a key at the wrong
    # indentation -- loads cleanly, does nothing, and leaves the operator
    # believing they configured something. Checked against config.yml and
    # config_default.yml before turning on: neither carries an unknown key, so
    # this refuses nothing that works today.
    model_config = ConfigDict(extra="forbid")

    features: Features
    # Optional, and None rather than a default instance: a config.yml with no
    # `llm:` section must behave exactly as it did before this field existed
    # (spec 003 FR-002), and "absent" has to be distinguishable from "present
    # and empty" for that to hold.
    llm: LLMConfig | None = None
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
        """Load config.yml, or the shipped defaults when there is no config.yml.

        A *present but invalid* config.yml raises. It used to be swallowed, and
        both of the quiet options are wrong:

        - returning None disables every config-driven feature including the
          message quota, so one typo removed rate limiting for everybody;
        - falling back to config_default.yml silently applies settings nobody
          chose -- it would re-enable `postprocessing` (external web search, and
          its per-message cost) for an operator who had deliberately turned it
          off, and replace their quota with the default 100.

        Refusing to start is the only option that cannot quietly do the wrong
        thing: the typo surfaces at deploy time rather than in a bill. An absent
        config.yml is a different case and still falls back, because running
        with documented defaults is what a fresh checkout expects.
        """
        if config_yml != CONFIG_DEFAULT_YML:
            try:
                return cls._load(config_yml)
            except (FileNotFoundError, IsADirectoryError):
                # docker-compose bind-mounts ./config.yml; when the host file is
                # missing Docker creates a directory in its place, so both mean
                # "no config supplied".
                logging.warning(
                    f"No config at {config_yml}; using {CONFIG_DEFAULT_YML}"
                )
            except (ValidationError, ValueError, yaml.YAMLError) as e:
                raise SystemExit(
                    f"Invalid config {config_yml}:\n{e}\n\n"
                    "Refusing to start. Fix the file, or remove it to run with "
                    f"the defaults in {CONFIG_DEFAULT_YML}."
                ) from e

        try:
            return cls._load(CONFIG_DEFAULT_YML)
        except Exception as e:
            raise SystemExit(
                f"The shipped default config {CONFIG_DEFAULT_YML} is unusable, "
                f"which should not happen in a working checkout:\n{e}"
            ) from e
