# From V-GPS (https://github.com/nakamotoo/V-GPS), jaxrl_m of bridge_data_v2 (https://github.com/rail-berkeley/bridge_data_v2), MIT. Modified for this thesis.
from .continuous.bc import BCAgent
from .continuous.calql import CalQLAgent
from .continuous.cql import ContinuousCQLAgent
from .continuous.gc_bc import GCBCAgent
from .continuous.gc_ddpm_bc import GCDDPMBCAgent
from .continuous.gc_iql import GCIQLAgent
from .continuous.iql import IQLAgent
from .continuous.sac import SACAgent
from .continuous.sarsa import SARSAAgent
from .continuous.sarsa_ensemble import SARSAEnsembleAgent
from .continuous.sarsa_ensemble_siglip import SigLIPSARSAEnsembleAgent
from .binary_classifier import BinaryClassifierAgent

agents = {
    "gc_bc": GCBCAgent,
    "gc_iql": GCIQLAgent,
    "gc_ddpm_bc": GCDDPMBCAgent,
    "bc": BCAgent,
    "iql": IQLAgent,
    "cql": ContinuousCQLAgent,
    "calql": CalQLAgent,
    "sac": SACAgent,
    "sarsa": SARSAAgent,
    "sarsa_ensemble": SARSAEnsembleAgent,
    "sarsa_ensemble_siglip": SigLIPSARSAEnsembleAgent,
    "binary_classifier": BinaryClassifierAgent,
}
