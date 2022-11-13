import numpy as np
from mujoco_worldgen import Env, WorldParams, WorldBuilder, Floor, ObjFromXML, Geom
import random

class MetaRLEnv():
	def __init__(self, 
		floor_size = 5., 
		world_height = 2.5,
		wall_thickness = 0.01,
		wall_height = 2.5/ 4,
		cube_size = 0.25,
		num_cube_range=(5, 10),
		seed=4,
	) -> None:
		self.floor_size=floor_size
		self.world_height=world_height
		self.wall_thickness=wall_thickness
		self.wall_height=wall_height
		self.cube_size = cube_size
		self.num_cube_range = num_cube_range
		self.seed = seed
		self.cube_size = cube_size

		self.world_params = WorldParams(size=(floor_size, floor_size, world_height))
		self.builder = WorldBuilder(self.world_params, self.seed)
		self.floor = Floor()
		self.builder.append(self.floor)
	
	def _make_wall(self, wall_shape, placement_xy, name):
		wall = Geom('box', wall_shape, name=name)
		wall.mark_static()
		self.floor.append(wall, placement_xy=placement_xy)

	def _make_room(self):
		wall_thickness, floor_size, wall_height = self.wall_thickness, self.floor_size, self.wall_height
		self._make_wall((wall_thickness, floor_size, wall_height), (0, 0), "wall1")
		self._make_wall((wall_thickness, floor_size, wall_height), (1, 0), "wall2")
		self._make_wall((floor_size - wall_thickness*2, wall_thickness, wall_height), (1/2, 0), "wall3")
		self._make_wall((floor_size - wall_thickness*2, wall_thickness, wall_height), (1/2, 1), "wall4")
	
	def _make_player(self):
		player = ObjFromXML("particle_hinge")
		self.floor.append(player) #place player at random location
		player.mark("player")

	def _make_cubes(self, num_cubes):
		for i in range(num_cubes):
			cube = Geom('box', self.cube_size, name=f"cube{i}")
			self.floor.append(cube)
			cube.mark(f"cube{i}")

	def get_sim(self, seed):
		random.seed(seed)
		self._make_room()
		self._make_player()

		self.num_cubes = random.randrange(self.num_cube_range[0], self.num_cube_range[1])
		self._make_cubes(self.num_cubes)

		self.floor.mark("target", (.5, .5, 0.05))
		return self.builder.get_sim()

	def get_reward(self, sim):
		#reward for one box in the middle
		target_xpos = sim.data.get_site_xpos("target")
		min_dist = 100
		for i in range(self.num_cubes):
			object_xpos = sim.data.get_site_xpos(f"cube{i}")
			dist = np.sum(np.square(object_xpos - target_xpos))
			if dist < min_dist: min_dist = dist
		return 1/min_dist
	

def make_env(num_cube_range=(5,10)):
	env = MetaRLEnv(num_cube_range=num_cube_range)
	return Env(
		get_sim=lambda seed: env.get_sim(seed), 
		get_reward=lambda sim: env.get_reward(sim), 
		horizon=30
	)


