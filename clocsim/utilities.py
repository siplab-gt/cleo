from collections.abc import MutableMapping

from brian2 import second
from brian2.groups.group import get_dtype
from brian2.equations.equations import (Equations, DIFFERENTIAL_EQUATION,
                                        SUBEXPRESSION, PARAMETER,
                                        check_subexpressions,
                                        extract_constant_subexpressions)

def modify_model_with_eqs(neuron_group, eqs_to_add):
    '''
    Adapted from _create_variables() from neurongroup.py from Brian2
    source code v2.3.0.2
    '''
    if type(eqs_to_add) == str:
        eqs_to_add = Equations(eqs_to_add)
    neuron_group.equations += eqs_to_add

    variables = neuron_group.variables

    dtype = {}
    if isinstance(dtype, MutableMapping):
        dtype['lastspike'] = neuron_group._clock.variables['t'].dtype


    for eq in eqs_to_add.values():
        dtype = get_dtype(eq, dtype)  # {} appears to be the default
        if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
            if 'linked' in eq.flags:
                # 'linked' cannot be combined with other flags
                if not len(eq.flags) == 1:
                    raise SyntaxError(('The "linked" flag cannot be '
                                       'combined with other flags'))
                neuron_group._linked_variables.add(eq.varname)
            else:
                constant = 'constant' in eq.flags
                shared = 'shared' in eq.flags
                size = 1 if shared else neuron_group._N
                variables.add_array(eq.varname, size=size,
                                         dimensions=eq.dim, dtype=dtype,
                                         constant=constant,
                                         scalar=shared)
        elif eq.type == SUBEXPRESSION:
            neuron_group.variables.add_subexpression(eq.varname, dimensions=eq.dim,
                                             expr=str(eq.expr),
                                             dtype=dtype,
                                             scalar='shared' in eq.flags)
        else:
            raise AssertionError('Unknown type of equation: ' + eq.eq_type)

    # Add the conditional-write attribute for variables with the
    # "unless refractory" flag
    if neuron_group._refractory is not False:
        for eq in neuron_group.equations.values():
            if (eq.type == DIFFERENTIAL_EQUATION and
                        'unless refractory' in eq.flags):
                not_refractory_var = neuron_group.variables['not_refractory']
                var = neuron_group.variables[eq.varname]
                var.set_conditional_write(not_refractory_var)

    # Stochastic variables
    for xi in neuron_group.equations.stochastic_variables:
        neuron_group.variables.add_auxiliary_variable(xi, dimensions=(second ** -0.5).dim)

    # Check scalar subexpressions
    for eq in neuron_group.equations.values():
        if eq.type == SUBEXPRESSION and 'shared' in eq.flags:
            var = neuron_group.variables[eq.varname]
            for identifier in var.identifiers:
                if identifier in neuron_group.variables:
                    if not neuron_group.variables[identifier].scalar:
                        raise SyntaxError(('Shared subexpression %s refers '
                                           'to non-shared variable %s.')
                                          % (eq.varname, identifier))

        