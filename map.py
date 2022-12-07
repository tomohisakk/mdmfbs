import queue
import random
import collections
import numpy as np

class Symbols():
	State = "D"
	Goal = "G"
	Static_module = "#"
	Dynamic_module = "*"
	Health = "."

class MakeMap():
	def __init__(self, w, h, dsize, s_modules, d_modules):
		super(MakeMap, self).__init__()
		assert w>0 and h>0 and dsize>0
		assert 0<=s_modules and 0<=d_modules
		self.w = w
		self.h = h
		self.dsize = dsize
		self.s_modules = s_modules
		self.d_modules = d_modules

		self.symbols = Symbols()
		self.map = self._make_map()

	def _make_map(self):
		map = np.random.choice([".", "#", '*'], (self.h, self.w), p=[1, 0, 0])

		for _ in range(self.s_modules):
			i = random.randint(0, self.w-1)
			j = random.randint(0, self.h-1)
			map[j][i] = '#'
		
		for _ in range(self.d_modules):
			i = random.randint(0, self.w-1)
			j = random.randint(0, self.h-1)
			map[j][i] = '*'

		i = 0

		# Set droplet
		map[0][0] = "D"
		map[-1][-1] = "D"

		self.map = map

	def _is_touching(self, state, obj):
		if self.map[state[1]][state[0]] == obj:
			return True
		return False

	def _is_map_good(self, start):
		queue = collections.deque([[start]])
		seen = set([start])
#		print(self.map)
		while queue:
			path = queue.popleft()
#			print(path)
			x, y = path[-1]
			if x==self.w-1 and y==self.h-1:
				return True
			for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
				if 0 <= x2 < (self.w-self.dsize+1) and 0 <= y2 < (self.h-self.dsize+1) and \
				(self._is_touching((x2,y2), self.symbols.Dynamic_module) == False) and\
				(self._is_touching((x2,y2), self.symbols.Static_module) == False) and\
				(x2, y2) not in seen:
					queue.append(path + [(x2, y2)])
					seen.add((x2, y2))
#		print("Bad map")
#		print(self.map)
		return False

	def gen_random_map(self):
		self._make_map()
		while self._is_map_good((0,0)) == False:
			self._make_map()
		return self.map

"""
if __name__ == '__main__':
	mapclass = MakeMap(w=8, h=8, dsize=1, s_modules=3, d_modules=0)
	while True:
		map = mapclass.gen_random_map()
		print(map)
"""