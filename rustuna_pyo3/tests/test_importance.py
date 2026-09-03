from __future__ import annotations

from collections.abc import Callable

import pytest
from _pytest.fixtures import SubRequest
from optuna.importance import BaseImportanceEvaluator
from optuna.testing.pytest_importance import (
    BasicImportanceEvaluatorTestCase,
    ConditionalImportanceEvaluatorTestCase,
    MultiObjectiveImportanceEvaluatorTestCase,
)

from rustuna.converter import ToOptunaImportanceEvaluator
from rustuna.importance import PedAnovaImportanceEvaluator


class TestBasicImportanceEvaluator(BasicImportanceEvaluatorTestCase):
    @pytest.fixture(params=[PedAnovaImportanceEvaluator])
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return lambda: ToOptunaImportanceEvaluator(request.param())


class TestConditionalImportanceEvaluator(ConditionalImportanceEvaluatorTestCase):
    @pytest.fixture(params=[PedAnovaImportanceEvaluator])
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return lambda: ToOptunaImportanceEvaluator(request.param())


class TestMultiObjectiveImportanceEvaluator(MultiObjectiveImportanceEvaluatorTestCase):
    @pytest.fixture(params=[PedAnovaImportanceEvaluator])
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return lambda: ToOptunaImportanceEvaluator(request.param())
