from pandac.PandaModules import *
from math import floor

class Agent_Controller():
	dx = [0,1,0,-1]
	dy = [-1,0,1,0]
	TURN_LEFT = 0
	TURN_RIGHT = 1
	MOVE_FORWARD = 2
	STOP = 3

	def __init__(self, src, dest):
		self.src = src
		self.dest = dest
		self.curr = self.src
		self.curr_cube = self.p2c(self.curr)
		self.dest_cube = self.p2c(self.dest)
		self.to_cube = LPoint3f(self.curr_cube)

		print self.src, "src"
		print self.curr_cube, "curr"
		print self.to_cube, "to"

		self.visited = [[0 for j in range(101)] for i in range(101)]
		self.dfs_stack = []

	def p2c(self, pos):
		return LPoint3f(floor(pos[0]/0.5),floor(pos[1]/0.5),floor(pos[2]/0.5))


	def get_next_action(self, pos, object_dirn, left_ram, right_ram):
		self.curr = pos
		self.curr_cube = self.p2c(self.curr)

		next_move = ''

		if self.to_cube == self.curr_cube:
			self.generate_next_cube()
		elif self.to_cube == self.dest_cube:
			next_move = self.STOP
			return next_move

		to_dirn = LVector3f(self.to_cube.x-self.curr_cube.x, self.to_cube.y-self.curr_cube.y, self.to_cube.z-self.curr_cube.z)
		alignment = object_dirn.relativeAngleDeg(to_dirn)

		print alignment
		if abs(alignment) <= 1e-5:
			next_move = self.MOVE_FORWARD
		elif alignment < 0:
			next_move = self.TURN_RIGHT
		else:
			next_move = self.TURN_LEFT

		return next_move

	def check_visited(self,x,y):
		return self.visited[x+50][y+50]

	def set_visited(self,x,y,val):
		self.visited[x+50][y+50] = val

	def generate_next_cube(self):
		self.dfs_stack.append((self.to_cube.x,self.to_cube.y))
		self.set_visited(int(self.to_cube.x),int(self.to_cube.y),1)

		prospects = []

		for i in range(4):
			xx = int(self.to_cube.x+self.dx[i])
			yy = int(self.to_cube.y+self.dy[i])
			prospects.append((xx,yy,abs(xx-self.dest_cube.x) + abs(yy-self.dest_cube.y) ))

		prospects = sorted(prospects, key=lambda prospect:prospect[2])

		for (xx,yy,t) in prospects:
			print (xx,yy,t)

		print "-----"

		for (xx,yy,t) in prospects:
			if self.check_visited(xx,yy) == 0:
				self.to_cube = LPoint3f(xx,yy,self.to_cube.z)
				return

		self.dfs_stack.pop()
		self.to_cube = LPoint3f(dfs_stack[-1][0],dfs_stack[-1][1],self.to_cube.z)


		


