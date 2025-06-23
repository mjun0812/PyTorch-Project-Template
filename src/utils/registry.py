# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# This code from fvcore (https://github.com/facebookresearch/fvcore)

from collections.abc import Iterator
from typing import Any

from tabulate import tabulate


class Registry:
    """Registry that provides name-to-object mapping for dynamic component discovery.

    This registry system supports third-party users' custom modules by allowing
    dynamic registration and retrieval of classes and functions.

    Examples:
        Creating a registry:

        >>> BACKBONE_REGISTRY = Registry('BACKBONE')

        Registering an object using decorator:

        >>> @BACKBONE_REGISTRY.register()
        ... class MyBackbone():
        ...     pass

        Registering an object using function call:

        >>> BACKBONE_REGISTRY.register(MyBackbone)

        Getting an object from registry:

        >>> backbone_cls = BACKBONE_REGISTRY.get("MyBackbone")
    """

    def __init__(self, name: str) -> None:
        """Initialize the registry.

        Args:
            name: The name identifier for this registry.
        """
        self._name: str = name
        self._obj_map: dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        """Internal method to register an object.

        Args:
            name: Name to register the object under.
            obj: Object to register.

        Raises:
            AssertionError: If an object with the same name is already registered.
        """
        assert name not in self._obj_map, (
            f"An object named '{name}' was already registered in '{self._name}' registry!"
        )
        self._obj_map[name] = obj

    def register(self, obj: object | None = None, name: str | None = None) -> object | None:
        """Register an object under the given name.

        Can be used as either a decorator or function call.

        Args:
            obj: Object to register. If None, returns a decorator.
            name: Name to register the object under. If None, uses obj.__name__.

        Returns:
            The registered object or a decorator function.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object, name: str | None = None) -> object:
                if name is None:
                    register_name = func_or_class.__name__  # pyre-ignore
                else:
                    register_name = name
                self._do_register(register_name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            register_name = obj.__name__  # pyre-ignore
        else:
            register_name = name
        self._do_register(register_name, obj)

    def get(self, name: str) -> object:
        """Get an object from the registry.

        Args:
            name: Name of the object to retrieve.

        Returns:
            The registered object.

        Raises:
            KeyError: If no object with the given name is found in the registry.
        """
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered in the registry.

        Args:
            name: Name to check.

        Returns:
            True if the name is registered, False otherwise.
        """
        return name in self._obj_map

    def __repr__(self) -> str:
        """Return a string representation of the registry.

        Returns:
            Formatted table showing all registered names and objects.
        """
        table_headers = ["Names", "Objects"]
        table = tabulate(self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid")
        return f"Registry of {self._name}:\n" + table

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Iterate over all registered name-object pairs.

        Returns:
            Iterator yielding (name, object) tuples.
        """
        return iter(self._obj_map.items())
