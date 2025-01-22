import re
from fnmatch import fnmatch


def match_user(users_spec: list[str] | None, user_id: str | None) -> bool:
    if not users_spec:
        return True
    for entry in users_spec:
        if entry == "all":
            return True
        if user_id is None:
            if entry == "guests":
                return True
        else:
            if entry == "logged_in":
                return True
            elif entry[0] == "/" and entry[-1] == "/":
                if re.search(entry[1:-1], user_id):
                    return True
            else:
                if fnmatch(user_id, entry):
                    return True
    return False
