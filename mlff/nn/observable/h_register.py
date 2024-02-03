from .observable import Energy
from .observable_sparse import EnergySparse
from .observable_sparse import DipoleSparse
from .observable_sparse import HirshfeldSparse


def get_observable_module(name, h):
    if name == 'energy':
        return Energy(**h)
    elif name == 'energy_sparse':
        return EnergySparse(**h)
    elif name == 'dipole_sparse':
        return DipoleSparse(**h)    
    elif name == 'hirsh_ratios_sparse':
        return HirshfeldSparse(**h)  
    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
