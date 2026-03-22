import logging
from typing import Any, Dict, List, MutableMapping, Tuple

from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Transformer,
)
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from .config import DistributionConfig, DistributionType

log = logging.getLogger(__name__)


def create_optuna_distribution_from_config(
    config: MutableMapping[str, Any],
) -> BaseDistribution:
    kwargs = dict(config)
    if isinstance(config["type"], str):
        kwargs["type"] = DistributionType[config["type"]]
    param = DistributionConfig(**kwargs)

    if param.type == DistributionType.categorical:
        assert param.choices is not None
        return CategoricalDistribution(param.choices)

    if param.type == DistributionType.int:
        assert param.low is not None
        assert param.high is not None
        step = int(param.step) if param.step is not None else 1
        return IntDistribution(
            int(param.low), int(param.high), log=param.log, step=step
        )

    if param.type == DistributionType.float:
        assert param.low is not None
        assert param.high is not None
        if param.log:
            return FloatDistribution(param.low, param.high, log=True)
        if param.step is not None:
            return FloatDistribution(param.low, param.high, step=param.step)
        return FloatDistribution(param.low, param.high)

    raise NotImplementedError(f"{param.type} is not supported by Optuna sweeper.")


def create_optuna_distribution_from_override(override: Override) -> Any:
    if not override.is_sweep_override():
        return override.get_value_element_as_str()

    value = override.value()
    choices: List[CategoricalChoiceType] = []

    if override.is_choice_sweep():
        assert isinstance(value, ChoiceSweep)
        for x in override.sweep_iterator(transformer=Transformer.encode):
            assert isinstance(
                x, (str, int, float, bool, type(None))
            ), f"A choice sweep expects str, int, float, bool, or None type. Got {type(x)}."
            choices.append(x)
        return CategoricalDistribution(choices)

    if override.is_range_sweep():
        assert isinstance(value, RangeSweep)
        assert value.start is not None
        assert value.stop is not None
        if value.shuffle:
            for x in override.sweep_iterator(transformer=Transformer.encode):
                assert isinstance(
                    x, (str, int, float, bool, type(None))
                ), f"A choice sweep expects str, int, float, bool, or None type. Got {type(x)}."
                choices.append(x)
            return CategoricalDistribution(choices)
        if (
            isinstance(value.start, float)
            or isinstance(value.stop, float)
            or isinstance(value.step, float)
        ):
            return FloatDistribution(value.start, value.stop, step=value.step)
        return IntDistribution(
            int(value.start), int(value.stop), step=int(value.step)
        )

    if override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)
        assert value.start is not None
        assert value.end is not None
        if "log" in value.tags:
            if isinstance(value.start, int) and isinstance(value.end, int):
                return IntDistribution(int(value.start), int(value.end), log=True)
            return FloatDistribution(value.start, value.end, log=True)
        else:
            if isinstance(value.start, int) and isinstance(value.end, int):
                return IntDistribution(value.start, value.end)
            return FloatDistribution(value.start, value.end)

    raise NotImplementedError(f"{override} is not supported by Optuna sweeper.")


def create_params_from_overrides(
    arguments: List[str],
) -> Tuple[Dict[str, BaseDistribution], Dict[str, Any]]:
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(arguments)
    search_space_distributions: Dict[str, BaseDistribution] = {}
    fixed_params: Dict[str, Any] = {}

    for override in parsed:
        param_name = override.get_key_element()
        value = create_optuna_distribution_from_override(override)
        if isinstance(value, BaseDistribution):
            search_space_distributions[param_name] = value
        else:
            fixed_params[param_name] = value
    return search_space_distributions, fixed_params
