from pandac.PandaModules import *
from math import floor
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import cv2

from training import pred_fn,offset,scale

from time import sleep

class Agent_Controller():
    # dx = [0,1,0,-1]
    # dy = [-1,0,1,0]
    dx = [0,1,0,-1,1,1,-1,-1]
    dy = [-1,0,1,0,-1,1,-1,1]
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

        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=5)
        print self.src, "src"
        print self.curr_cube, "curr"
        print self.dest_cube, "dest"

        self.visited = [[0 for j in range(101)] for i in range(101)]
        self.dfs_stack = []
        # plt.ion()
        # plt.figure(figsize=(5,5))

    def p2c(self, pos):
        return LPoint3f(floor(pos[0]/0.5),floor(pos[1]/0.5),floor(pos[2]/0.5))


    def get_next_action(self, pos, object_dirn, left_ram_ptr, right_ram_ptr):
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
        print self.to_cube,'to_cube'
        print self.curr_cube,'curr_cube'

        # print alignment
        if abs(alignment) <= 1e-3:
            depth_m = self.get_depth_map(left_ram_ptr,right_ram_ptr)
            if self.possible_fow(depth_m):
                return self.MOVE_FORWARD
            else:
                self.set_visited(self.to_cube.x,self.to_cube.y,1)
                self.generate_new_cube()
                return None
        elif alignment < 0:
            # sleep(0.05)
            next_move = self.TURN_RIGHT
        else:
            # sleep(0.05)
            next_move = self.TURN_LEFT
        return next_move

    def check_visited(self,x,y):
        return self.visited[x+50][y+50]

    def set_visited(self,x,y,val):
        x,y = int(x),int(y)
        self.visited[x+50][y+50] = val

    def generate_next_cube(self):
        self.dfs_stack.append((self.to_cube.x,self.to_cube.y))
        self.set_visited(int(self.to_cube.x),int(self.to_cube.y),1)

        prospects = []

        for i in range(8):
            xx = int(self.to_cube.x+self.dx[i])
            yy = int(self.to_cube.y+self.dy[i])
            prospects.append((xx,yy,(xx-self.dest_cube.x)**2 + (yy-self.dest_cube.y)**2))
            # prospects.append((xx,yy,abs(xx-self.dest_cube.x) + abs(yy-self.dest_cube.y)))

        prospects = sorted(prospects, key=lambda prospect:prospect[2])

        # for (xx,yy,t) in prospects:
        #   print (xx,yy,t)

        # print "-----"

        for (xx,yy,t) in prospects:
            if self.check_visited(xx,yy) == 0:
                self.to_cube = LPoint3f(xx,yy,self.to_cube.z)
                return

        self.dfs_stack.pop()
        self.to_cube = LPoint3f(dfs_stack[-1][0],dfs_stack[-1][1],self.to_cube.z)

    def generate_new_cube(self):

        prospects = []

        for i in range(8):
            xx = int(self.curr_cube.x+self.dx[i])
            yy = int(self.curr_cube.y+self.dy[i])
            prospects.append((xx,yy,(xx-self.dest_cube.x)**2 + (yy-self.dest_cube.y)**2))            
            # prospects.append((xx,yy,abs(xx-self.dest_cube.x) + abs(yy-self.dest_cube.y)))

        prospects = sorted(prospects, key=lambda prospect:prospect[2])

        print '---- prospects'
        for (xx,yy,t) in prospects:
          print (xx,yy,t)

        print "-----"

        for (xx,yy,t) in prospects:
            if self.check_visited(xx,yy) == 0:
                self.to_cube = LPoint3f(xx,yy,self.to_cube.z)
                return



    def possible_fow(sefl,im):
        mean_depth = im[40:,80:120].mean()
        # mean_depth2 = im[140:180,80:120].mean()
        # print mean_depth,mean_depth2
        # if mean_depth > 0.95 or mean_depth2 > 0.95:
        if mean_depth > 0.85:
            return False
        else:
            return True

    def rgb2gray(self,rgb):
        shape = rgb.shape
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).reshape((shape[0],shape[1],1)).astype('float32')

    def get_depth_map(self,left_ram_ptr,right_ram_ptr):

        # left_inp = cv2.cvtColor(np.frombuffer(left_ram_ptr.get_data(),np.uint8).reshape(200,200,3),cv2.COLOR_RGB2GRAY)
        # right_inp = cv2.cvtColor(np.frombuffer(right_ram_ptr.get_data(),np.uint8).reshape(200,200,3),cv2.COLOR_RGB2GRAY)
        left_inp = np.frombuffer(left_ram_ptr.get_data(),np.uint8).reshape(200,200,3).astype('float32')
        right_inp = np.frombuffer(right_ram_ptr.get_data(),np.uint8).reshape(200,200,3).astype('float32')
        left_inp,right_inp = self.rgb2gray(left_inp),self.rgb2gray(right_inp)
        left_inp = np.flipud(left_inp)
        right_inp = np.flipud(right_inp)
        inp = np.concatenate((right_inp,left_inp),-1)
        inp = np.rollaxis(inp,2)
        inp = (inp-offset)/scale
        # print inp.shape
        out = pred_fn(inp.reshape((1,)+inp.shape))
        print out[0].shape
        # plt.clf()
        # disparity = self.stereo.compute(left_inp,right_inp)
        # plt.imshow(out[0].reshape(200,200),cmap='viridis')
        # plt.colorbar()
        # plt.savefig('fig.png')
        # plt.clf()
        # plt.imshow(left_inp.reshape(200,200))
        # plt.savefig('fig1.png')
        # plt.imshow(right_inp.reshape(200,200))
        # plt.savefig('fig2.png')
        # plt.draw()
        # print left_inp.shape
        # plt.imshow(left_inp)
        return out[0].reshape(200,200)



