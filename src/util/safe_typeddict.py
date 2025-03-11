from typing import Any, cast


class _SafeTypedDictRuntime(dict):
    """
    An alternative runtime class for TypedDict that avoids KeyErrors.
    By default,
        1) TypedDict behaves as a regular dict and
        2) Type-checkers can't guarantee TypedDict totality in all scenarios.
    This is a subclass of dict that tracks TypedDict annotations at runtime
        to provide type-safe default values for unset dict keys.
    Do not use this class directly, instead use the @safe_typeddict decorator
        on a TypedDict subclass.
    """

    _td_annotations: dict[str, type]

    def __getitem__(self, key: Any, /) -> Any:
        return super().get(key, self._td_annotations[key]())

    def __init_subclass__(cls) -> None:
        bases = cast(tuple[_SafeTypedDictRuntime, ...], cls.__bases__)
        cls._td_annotations = {}
        for base in bases:  # collect inherited annotations
            cls._td_annotations.update(base._td_annotations)
        cls._td_annotations.update(cls.__annotations__)

    @classmethod
    def use[TD: type](cls, td: TD) -> TD:
        """Decorator to be used on TypedDict subclasses"""
        bases: tuple[type, ...] = (cls,) + tuple(
            b for b in td.__bases__ if issubclass(b, cls)
        )
        subcls: TD = type(
            td.__name__,
            bases,
            td.__dict__.copy(),
        )  # type: ignore[assignment]
        return subcls


safe_typeddict = _SafeTypedDictRuntime.use
