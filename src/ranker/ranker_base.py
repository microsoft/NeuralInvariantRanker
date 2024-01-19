# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import sys

if True:
    PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../../../"))
    if PROJECT_PATH not in sys.path:
        sys.path.append(PROJECT_PATH)
    from src.config import Config


class BaseRanker:
    def __init__(
        self,
        config: Config,
        *args,
        **kwargs
    ):
        self.config = config

    def rank(
        self,
        problem: str,
        invariants: List[str],
        varifiction_arguments: Optional[Dict[str, Any]] = None,
        verification_function: Optional[Callable] = None,
        verification_results: Optional[List[Tuple[bool, Any, Dict[str, Any]]]] = None,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError("Child classes must implement this")

    def score(
        self,
        problem: str,
        invariant: List[str],
        varifiction_arguments: Optional[Dict[str, Any]] = None,
        verification_function: Optional[Callable] = None,
        verification_result: Optional[Tuple[bool, Any, Dict[str, Any]]] = None,
    ) -> float:
        raise NotImplementedError("Child classes must implement this")