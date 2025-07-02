"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
            self.sim.model.body_mass[1] *= 0.7

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())


    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""
        #
        # TASK 6: implement domain randomization. Remember to sample new dynamics parameter
        # at the start of each training episode.
        
        # Original masses: thigh, leg, foot
        # Assume some default ranges (tune them manually later)
        thigh_mass_range = (0.5, 1.5)*self.original_masses[2]
        leg_mass_range = (0.5, 1.5)*self.original_masses[3]
        foot_mass_range = (0.5, 1.5)*self.original_masses[4]

        thigh_mass = np.random.uniform(*thigh_mass_range)
        leg_mass = np.random.uniform(*leg_mass_range)
        foot_mass = np.random.uniform(*foot_mass_range)

        # Torso mass is fixed and not randomized; we retrieve the current (possibly scaled) value
        torso_mass = self.sim.model.body_mass[1]

        # Return all masses in order: torso, thigh, leg, foot
        return np.array([torso_mass, thigh_mass, leg_mass, foot_mass])

        

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = self.sample_parameters(self)

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3* np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

#aggiungere una funzione per resettare il model con udr e tenere questa per resettare il modello per i primi 2 algoritmi
    def reset_model(self):
        """Reset the environment to a random initial state"""
        #self.set_random_parameters() #Applichiamo UDR
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments

"""
#qui sono definiti i vari envs che poi si vanno a usare nel training
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

