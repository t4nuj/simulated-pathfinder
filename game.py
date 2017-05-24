from agent_controller import Agent_Controller
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
from time import sleep
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import sys
# from panda3d.core import CollisionNode


class Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        loadPrcFileData("", "win-size 400 200")
        loadPrcFileData("", "show-buffers t")
        winProp = WindowProperties()
        winProp.setSize(800,800)
        base.win.requestProperties(winProp)

        self.cTrav = CollisionTraverser()
        self.chandler = CollisionHandlerQueue()

        # Set room
        self.room = loader.loadModel('custom/squareRoom/squareRoom')
        self.room.reparentTo(self.render)
        self.room.setPos(0, 0, 0)

        # Set light source
        plight = PointLight('plight')
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(0, 0, 4)
        self.render.setLight(plnp)

        plight = PointLight('plight2')
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(-5, -5, 5)
        self.render.setLight(plnp)

        plight = PointLight('plight3')
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(5, -5, 5)
        # self.render.setLight(plnp)

        # plight = PointLight('plight4')
        # plnp = self.render.attachNewNode(plight)
        # plnp.setPos(-5, 5, 5)
        # self.render.setLight(plnp)

        # plight = PointLight('plight5')
        # plnp = self.render.attachNewNode(plight)
        # plnp.setPos(5, 5, 5)
        # self.render.setLight(plnp)

        # Load agent model
        self.agent = loader.loadModel('custom/agent.egg')
        self.agent.reparentTo(self.render)
        self.agent.setPos(5,1, 0.2)

        self.dest = loader.loadModel('custom/dest.egg')
        self.dest.reparentTo(self.render)
        self.dest.setScale(0.001)
        self.dest.setPos(-3,-1,0.2)


        # Do some collision sphere shit
        cs = CollisionSphere(0, 0, 0, 0.15)
        self.cagent = self.agent.attachNewNode(CollisionNode('cnode'))
        self.cagent.node().addSolid(cs)
        self.cTrav.addCollider(self.cagent, self.chandler)
        self.cTrav.traverse(self.render)

        min, max = self.agent.getTightBounds()
        print min, max

        min, max = self.room.getTightBounds()
        print min,max

        base.camNode.setActive(0)
        
        # self.cam_god = base.makeCamera(base.win)
        self.cam_god = base.makeCamera(base.win, displayRegion=(0,1,0.29,1))
        self.cam1 = base.makeCamera(base.win, displayRegion=(0.24,.49,0,0.288))
        self.cam2 = base.makeCamera(base.win, displayRegion=(.51,0.76,0,0.288))


        self.cam1.reparentTo(self.agent)
        self.cam2.reparentTo(self.agent)
        self.cam_god.reparentTo(self.agent)

        self.cam1.setPos(-0.03,0,0.15)
        self.cam2.setPos(0.03,0,0.15)
        self.cam_god.setPos(0,-10,10)
        self.cam1.node().getLens().setNear(0.05)
        self.cam2.node().getLens().setNear(0.05)
        # sel.cam_god.setAngle()
        self.cam_god.look_at(self.agent)


        self.left_dr = self.cam1.node().getDisplayRegion(0)
        self.right_dr = self.cam2.node().getDisplayRegion(0)

       
        # Keyboard bindings
        base.accept('w', self.go_forward)
        base.accept('s', self.go_backward)
        base.accept('a', self.turn_left)
        base.accept('d', self.turn_right)
        base.accept('u', self.go_up)
        base.accept('j', self.go_down)
        base.accept('r', self.register_tick)
        base.accept('t',self.deregister_tick)

        self.agent_cont = Agent_Controller(self.agent.getPos(), self.dest.getPos())
        # Task bindings
        self.count = 0


    def tick(self, task):

        left_tex = self.left_dr.getScreenshot()
        right_tex = self.right_dr.getScreenshot()
        if left_tex == None or right_tex == None:
            # camera hasn't loaded yet
            return task.cont
        if left_tex.getXSize() != 200 or right_tex.getXSize() != 200 or left_tex.getYSize() != 200 \
            or right_tex.getYSize() != 200:
            #window hasn't loaded yet
            return task.cont
        left_ram = left_tex.getRamImageAs('RGB')
        right_ram = right_tex.getRamImageAs('RGB')

        
        # print 'dim_cam',self.left_dr.getPixelHeight(),self.left_dr.getPixelWidth()
        
        object_dirn = self.render.getRelativeVector(self.agent, Vec3.forward())
        move = self.agent_cont.get_next_action(self.agent.getPos(), object_dirn, left_ram, right_ram)
        if move == Agent_Controller.TURN_LEFT:
            # print 'turn left'
            self.turn_left()
        elif move == Agent_Controller.TURN_RIGHT:
            # print 'turn right'
            # self.count += 1
            self.turn_right()
        elif move == Agent_Controller.STOP:
            print self.count
            return task.done
        elif move == Agent_Controller.MOVE_FORWARD:
            # print 'move_forward'
            # self.count += 1
            self.go_forward()
        else:
            pass

        # sleep(0.1)
        return task.cont

    def go_forward(self):

        # print self.agent.getPos()
        self.count += 1 

        self.agent.setPos(self.agent, 0, 0.1, 0)
        self.cTrav.traverse(self.render)
        for entry in self.chandler.getEntries():
            print entry
            print self.cam1.getPos()+self.agent.getPos()
            sys.exit()
        # print self.chandler.getNumEntries()
        print '---------------'
        # if self.chandler.getNumEntries() > 0:
            # self.agent.setPos(self.agent, 0, -0.1, 0)
        # print self.agent.getPos()
        # print "-----------------"


    def go_backward(self):
        # print self.agent.getPos()
        self.agent.setPos(self.agent, 0, -0.1, 0)
        self.cTrav.traverse(self.render)
        # print self.chandler.getNumEntries()
        if self.chandler.getNumEntries() > 0:
            self.agent.setPos(self.agent, 0, 0.1, 0)
        # print "-----------------"


    def turn_left(self):
        self.count += 1 
        self.agent.setH(self.agent.getH() + 5)

    def turn_right(self):
        self.count += 1 
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

    def register_tick(self):
        taskMgr.add(self.tick, "tick")

    def deregister_tick(self):
        pass




game = Game()
game.run()