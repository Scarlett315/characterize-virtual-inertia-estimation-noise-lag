"""
A pert file template.
"""
import traceback
import numpy as np

rng = np.random.default_rng(42)
_state = {'last_t': None, 
          'ou_vd': 0.0, 
          'ou_vq': 0.0,
          'ou_id': 0.0, 
          'ou_iq': 0.0}


SIGMA_V, SIGMA_I, TAU_OU = 0.008, 0.02, 0.2
STEP_TIME = 0.01 # 10 ms
#SIGMA_V, SIGMA_I, TAU_OU = 0.8, 0.8, 0.2  # deviations


# Orenstein-Uhlenbeck Simulation 
def ou_step(x, dt, tau, sigma):
    decay = np.exp(-dt / tau)
    return x * decay + sigma * np.sqrt(1 - decay**2) * rng.standard_normal()

# perturb!
def pert(t, system):
    """

    Perturbation function called at each step.

    The function needs to be named ``pert`` and takes two positional arguments:
    ``t`` for the simulation time, and ``system`` for the system object.
    Arbitrary logic and calculations can be applied in this function to
    ``system``.

    If the perturbation event involves switching, such as disconnecting a line,
    one will need to set the ``system.TDS.custom_event`` flag to ``True`` to
    trigger a system connectivity checking, and Jacobian rebuilding and
    refactorization. To implement, add the following line to the scope where the
    event is triggered:

    .. code-block :: python

        system.TDS.custom_event = True

    In other scopes of the code where events are not triggered, do not add the
    above line as it may cause significant slow-down.

    The perturbation file can be supplied to the CLI using the ``--pert``
    argument or supplied to :py:func:`andes.main.run` using the ``pert``
    keyword.

    Parameters
    ----------
    t : float
        Simulation time.
    system : andes.system.System
        System object supplied by the simulator.

    """
    try: 
        uid = 1

        grid_t = round(t / STEP_TIME) * STEP_TIME
        if _state.get('last_grid_t') == grid_t:
            return                      # already processed this real step
        _state['last_grid_t'] = grid_t
        dt = STEP_TIME                  # known exactly, since steps are fixed

        print(f"pert() applied at grid_t={grid_t}", flush=True)
        
        # ------------------ noise computation -------------------
        Tr = system.REGF2.get(src='Tr', attr='v', idx=uid)


        # compute OU noise for each var
        for k, sigma in [('ou_vd', SIGMA_V), 
                        ('ou_vq', SIGMA_V), 
                        ('ou_id', SIGMA_I), 
                        ('ou_iq', SIGMA_I)]:
            
            _state[k] = ou_step(_state[k], dt, TAU_OU, sigma)

        # add computed noise
        vd_n = system.REGF2.get(src='vd', attr='v', idx=uid)+ _state['ou_vd']
        vq_n = system.REGF2.get(src='vq', attr='v', idx=uid)+ _state['ou_vq']
        Id_n = system.REGF2.get(src='Id', attr='v', idx=uid)+ _state['ou_id']
        Iq_n = system.REGF2.get(src='Iq', attr='v', idx=uid)+ _state['ou_iq']

        # set variables
        """
        system.REGF2.set(src='vd', attr='v', idx=uid, value=vd_n)
        system.REGF2.set(src='vq', attr='v', idx=uid, value=vq_n)
        system.REGF2.set(src='Id', attr='v', idx=uid, value=Id_n)
        system.REGF2.set(src='Iq', attr='v', idx=uid, value=Iq_n)
        """

        
        # calculate noise on Psen_y
        Pe_noisy = Id_n * vd_n + Iq_n * vq_n

        decay = np.exp(-dt / Tr)
        old_y = system.REGF2.get(src='Pe', attr='v', idx=uid)


        # ---------------- manually calculate Psen_y -------------------
        Psen_y_noise = np.clip(Pe_noisy * (1 - decay), -0.5, 0.5)
        Psen_y_noisy = old_y * decay + Psen_y_noise
        print(f"Old value: {old_y}", flush=True)
        print(f"Psen_y_noise: {Psen_y_noise}", flush=True)
        print(f"state: {_state}")

        system.REGF2.set(src='Psen_y', attr='v', idx=uid, value=Psen_y_noisy)

    except Exception as e: 
        traceback.print_exc()

