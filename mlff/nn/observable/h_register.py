from .observable import Energy
from .observable_sparse import EnergySparse
from .observable_sparse import DipoleSparse
from .observable_sparse import DipoleVecSparse
from .observable_sparse import HirshfeldSparse
from .observable_sparse import DummySparse


def get_observable_module(name, h):
    if name == 'energy':
        return Energy(**h)
    elif name == 'energy_sparse':
        return EnergySparse(**h)
    elif name == 'dipole_sparse':
        return DipoleSparse(**h)    
    elif name == 'dipole_vec_sparse':
        return DipoleVecSparse(**h)    
    elif name == 'dummy_sparse':
        return DummySparse(**h)
    elif name == 'hirsh_ratios_sparse':
        return HirshfeldSparse(**h)  
    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)
