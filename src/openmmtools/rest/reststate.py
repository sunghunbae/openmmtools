from openmmtools.states import GlobalParameterState

import numpy as np


class RESTState(GlobalParameterState):
    lambda_rest_bonds = GlobalParameterState.GlobalParameter('lambda_rest_bonds', standard_value=1.0)
    lambda_rest_angles = GlobalParameterState.GlobalParameter('lambda_rest_angles', standard_value=1.0)
    lambda_rest_torsions = GlobalParameterState.GlobalParameter('lambda_rest_torsions', standard_value=1.0)
    lambda_rest_electrostatics = GlobalParameterState.GlobalParameter('lambda_rest_electrostatics', standard_value=0.0)
    lambda_rest_sterics = GlobalParameterState.GlobalParameter('lambda_rest_sterics', standard_value=0.0)

    def set_rest_parameters(self, beta_m, beta_0):
        """Set all defined lambda parameters to the given value.

        The undefined parameters (i.e. those being set to None) remain undefined.

        Parameters
        ----------
        new_value : float
            The new value for all defined parameters.
        """
        lambda_functions = {'lambda_rest_bonds': lambda beta_m, beta_0 : np.sqrt(beta_m / beta_0),
                 'lambda_rest_angles' : lambda beta_m, beta_0 : np.sqrt(beta_m / beta_0),
                 'lambda_rest_torsions' : lambda beta_m, beta_0 : np.sqrt(beta_m / beta_0),
                 'lambda_rest_electrostatics' : lambda beta_m, beta_0 : np.sqrt(beta_m / beta_0) - 1,
                 'lambda_rest_sterics' : lambda beta_m, beta_0 : beta_m / beta_0 - 1
                 }

        for parameter_name in self._parameters:
            if self._parameters[parameter_name] is not None:
                new_value = lambda_functions[parameter_name](beta_m, beta_0)
                setattr(self, parameter_name, new_value)