from enum import StrEnum, auto

from pydantic import BaseModel


class UserGroup(StrEnum):
    all = auto()
    logged_in = auto()


class Feature(BaseModel):
    enabled: bool
    user_group: UserGroup | None = None

    def matches_user_group(self, user_id: str | None) -> bool:
        if self.user_group == UserGroup.logged_in:
            return user_id is not None
        else:
            return True


class Features(BaseModel):
    postprocessing: Feature
