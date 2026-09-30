# From V-GPS (https://github.com/nakamotoo/V-GPS), jaxrl_m of bridge_data_v2 (https://github.com/rail-berkeley/bridge_data_v2), MIT. Unmodified.
from jaxrl_m.agents.continuous.cql import ContinuousCQLAgent


class CalQLAgent(ContinuousCQLAgent):
    """Same agent as CQL, just add an additional check that the use_calql flag is on."""

    @classmethod
    def create(
        cls,
        *args,
        **kwargs,
    ):
        kwargs["use_calql"] = True
        return super(CalQLAgent, cls).create(
            *args,
            **kwargs,
        )
