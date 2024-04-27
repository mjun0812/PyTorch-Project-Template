# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# This code from fvcore (https://github.com/facebookresearch/fvcore)

from typing import Any, Dict, Iterator, Optional, Tuple

from tabulate import tabulate


class Registry(object):
    """The registry that provides name -> object mapping,
    to support third-party users' custom modules.

    To create a registry (e.g. a backbone registry):
    ```python
    BACKBONE_REGISTRY = Registry('BACKBONE')
    ```

    To register an object:
    ```python
    @BACKBONE_REGISTRY.register()
    class MyBackbone():
    ```
    or
    ```python
    BACKBONE_REGISTRY.register(MyBackbone)

    To get an object from registry
    ```python
    BACKBONE_REGISTRY.get("MyBackbone")
    ```
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert (
            name not in self._obj_map
        ), f"An object named '{name}' was already registered in '{self._name}' registry!"
        self._obj_map[name] = obj

    def register(self, obj: object = None) -> Optional[object]:
        """Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of
        this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        """get object from Registry

        Args:
            name (str): Object Name

        Raises:
            KeyError: No object from registry

        Returns:
            object: Registred Object
        """
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid")
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())
