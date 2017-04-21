from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from pandac.PandaModules import loadPrcFileData
from panda3d.core import GraphicsWindow
from panda3d.core import WindowProperties



class Game(ShowBase):
	def __init__(self):
		ShowBase.__init__(self)

		loadPrcFileData("", "win-size 300 100")
		loadPrcFileData("", "show-buffers t")
		winProp = WindowProperties()
		winProp.setSize(300,100)
		base.win.requestProperties(winProp)

		environment = loader.loadModel('custom/Ground2/Ground2')
		environment.reparentTo(self.render)
		environment.setPos(0, 0, -1)

		room = loader.loadModel('custom/recliner/recliner')
		room.reparentTo(self.render)
		room.setPos(0, 20, 0)

		# base.cam.setPos (0,-20,1)


		self.agent = loader.loadModel('custom/sphere')
		self.agent.reparentTo(self.render)
		self.agent.setPos(0, 0, 1)
		self.agent.setScale(0.1)
		self.agent.setColor(100,0,0,1)

		base.camNode.setActive(0)

		self.cam1 = base.makeCamera(base.win, displayRegion=(0,.5,0,1))
		self.cam2 = base.makeCamera(base.win, displayRegion=(.5,1,0,1))

		self.cam1.reparentTo(self.agent)
		self.cam2.reparentTo(self.agent)

		self.cam1.setPos(-2,-20,0)
		self.cam2.setPos(2,-20,0)

		base.cam.reparentTo(self.agent)
		base.cam.setPos(0, -20, 0)

		self.x = 10
		self.y = 10
		self.z = 100
		
		base.accept('w', self.go_forward)
		base.accept('s', self.go_backward)
		base.accept('a', self.turn_left)
		base.accept('d', self.turn_right)
		base.accept('u', self.go_up)
		base.accept('j', self.go_down)


	def go_forward(self):
		self.agent.setPos(self.agent, 0, 5, 0)

	def go_backward(self):
		self.agent.setPos(self.agent, 0, -5, 0)

	def turn_left(self):
		self.agent.setH(self.agent.getH() + 5)

	def turn_right(self):
		self.agent.setH(self.agent.getH() - 5)

	def go_up(self):
		self.agent.setZ(self.agent.getZ() + 1)

	def go_down(self):
		self.agent.setZ(self.agent.getZ() - 1)


game = Game()
game.run()