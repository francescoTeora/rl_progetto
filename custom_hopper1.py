"""Implementation of the Hopper environment supporting domain randomization optimization.

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
#Qui salviamo i valori originali delle masse
        self.original_masses = np.copy(self.sim.model.body_mass[1:])  # [torso, thigh, leg, foot]
#Qui se ll'ambiente è source modifichiamo la massa del torso riducendola del 30%
        if domain == 'source':  # Torso mass is shifted -30% in the source domain
            self.sim.model.body_mass[1] *= 0.7
#Questi parametri controllano il Curriculum Domain Randomization (CDR):
#curriculum: attiva/disattiva la modalità.
#progress: indica quanto è avanzata la curriculum.
#episode_count: tiene traccia degli episodi.
#max_episodes: quante puntate dura il curriculum.
            
        # Curriculum DR settings
        self.curriculum = False
        self.progress = 0.0
        self.max_progress = 1.0  # Used for curriculum scaling
        self.episode_count = 0
        self.max_episodes = 1000  # Curriculum duration in episodes
        
#Chiama sample_parameters() per ottenere masse randomizzate e le applica con set_parameters()
    def set_random_parameters(self):
        """Set new randomized link masses."""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses using Uniform or Curriculum Domain Randomization."""
        # Torso mass stays fixed
        torso_mass = self.sim.model.body_mass[1]
#Praticamente se CDR è attivo, il range_factor (quanto randomizziamo) cresce gradualmente da 5% a 50%.
#Altrimenti si usa direttamente 50% di UDR.

        # Update curriculum progress
        if self.curriculum:
            self.episode_count += 1
            self.progress = min(self.episode_count / self.max_episodes, self.max_progress)
            range_factor = 0.05 + self.progress * 0.45  # Linearly grows from 5% to 50%
        else:
            range_factor = 0.5  # Default range for UDR

        randomized_masses = []
        for i in range(3):  # Only thigh, leg, foot (indices 0,1,2 after torso)
            default_mass = self.original_masses[i + 1]  # skip torso
            low = default_mass * (1 - range_factor)
            high = default_mass * (1 + range_factor)
            randomized_masses.append(np.random.uniform(low, high))
            
#Si randomizzano solo thigh, leg, foot.
#Per ciascuno, si genera un valore casuale tra ±range_factor rispetto al valore originale.
            
        return np.array([torso_mass] + randomized_masses)
#Ritorna l’array completo con tutte le masse in ordine 
    def enable_curriculum(self, active=True):
        """Enable or disable curriculum randomization."""
        self.curriculum = active

    def get_parameters(self):
        """Get the current masses of all relevant links."""
        return np.array(self.sim.model.body_mass[1:])

    def set_parameters(self, masses):
        """Set new values for all link masses."""
        self.sim.model.body_mass[1:] = masses

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset environment and apply new domain randomization."""
        self.set_random_parameters()
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
        mjstate = deepcopy(self.get_mujoco_state())
        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]
        self.set_sim_state(mjstate)

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_mujoco_state(self):
        return self.sim.get_state()


# Register environments
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
