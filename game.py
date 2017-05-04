from direct.showbase.ShowBase import ShowBase
from direct.task import Task
# from pandac.PandaModules import loadPrcFileData

# from pandac.PandaModules import CollisionSphere 
# from pandac.PandaModules import CollisionNode
# from pandac.PandaModules import CollisionTraverser
# from pandac.PandaModules import CollisionHandlerQueue
# from pandac.PandaModules import CollisionHandlerPusher
# from pandac.PandaModules import PNMImage
# from pandac.PandaModules import Filename

from pandac.PandaModules import *



from panda3d.core import GraphicsWindow
from panda3d.core import WindowProperties
from panda3d.core import PointLight
from panda3d.core import AmbientLight
from panda3d.core import VBase4

# from panda3d.core import CollisionNode

class Game(ShowBase):
	def __init__(self):
		ShowBase.__init__(self)

		loadPrcFileData("", "win-size 400 200")
		loadPrcFileData("", "show-buffers t")
		winProp = WindowProperties()
		winProp.setSize(400,200)
		base.win.requestProperties(winProp)

		self.cTrav = CollisionTraverser()
		self.chandler = CollisionHandlerQueue()

		# Set room
		self.room = loader.loadModel('custom/rooml/L')
		self.room.reparentTo(self.render)
		self.room.setPos(0, 0, 0)

		# Set light source
		plight = PointLight('plight')
		plnp = self.render.attachNewNode(plight)
		plnp.setPos(0, 0, 3)
		self.render.setLight(plnp)

		# Load agent model
		self.agent = loader.loadModel('custom/agent.egg')
		self.agent.reparentTo(self.render)
		self.agent.setPos(0, -2, 3)

		cs = CollisionSphere(0, 0, 0, 0.15)
		self.cagent = self.agent.attachNewNode(CollisionNode('cnode'))
		self.cagent.node().addSolid(cs)

		print self.agent.getPos()
		print self.cagent.getPos()

		self.cTrav.addCollider(self.cagent, self.chandler)
		self.cTrav.traverse(self.render)

		print self.chandler.getNumEntries()

		min, max = self.agent.getTightBounds()
		print min, max

		min, max = self.room.getTightBounds()
		print min,max

		print "--------"

		base.camNode.setActive(0)

		self.cam1 = base.makeCamera(base.win, displayRegion=(0,.5,0,1))
		self.cam2 = base.makeCamera(base.win, displayRegion=(.5,1,0,1))

		self.cam1.reparentTo(self.agent)
		self.cam2.reparentTo(self.agent)

		self.cam1.setPos(-0.03,0,0)
		self.cam2.setPos(+0.03,0,0)


		# Keyboard bindings
		base.accept('w', self.go_forward)
		base.accept('s', self.go_backward)
		base.accept('a', self.turn_left)
		base.accept('d', self.turn_right)
		base.accept('u', self.go_up)
		base.accept('j', self.go_down)

		# Task bindings
		taskMgr.add(self.task_go_to_destination, "task1")

	def take_frame_picture(self):
		p = PNMImage()
		base.win.getScreenshot(p)
		p.write(Filename("test5.jpg"))

	def task_go_to_destination(self, task):
		self.take_frame_picture()
		self.go_forward()
		return task.cont

	def go_forward(self):

		print self.agent.getPos()
		self.agent.setPos(self.agent, 0, 0.1, 0)
		self.cTrav.traverse(self.render)
		for entry in self.chandler.getEntries():
			print entry
		print self.chandler.getNumEntries()
		if self.chandler.getNumEntries() > 0:
			self.agent.setPos(self.agent, 0, -0.1, 0)
		print self.agent.getPos()
		print "-----------------"


	def go_backward(self):
		print self.agent.getPos()
		self.agent.setPos(self.agent, 0, -0.1, 0)
		self.cTrav.traverse(self.render)
		print self.chandler.getNumEntries()
		if self.chandler.getNumEntries() > 0:
			self.agent.setPos(self.agent, 0, 0.1, 0)
		print "-----------------"


	def turn_left(self):
		self.agent.setH(self.agent.getH() + 5)

	def turn_right(self):
		self.agent.setH(self.agent.getH() - 5)

	def go_up(self):
		self.agent.setPos(self.agent, 0, 0, 0.25)
		self.cTrav.traverse(self.render)
		if self.chandler.getNumEntries() > 0:
			self.agent.setPos(self.agent, 0, 0, -0.25)

	def go_down(self):
		self.agent.setPos(self.agent, 0, 0, -0.25)
		self.cTrav.traverse(self.render)
		if self.chandler.getNumEntries() > 0:
			self.agent.setPos(self.agent, 0, 0, 0.25)

game = Game()
game.run()