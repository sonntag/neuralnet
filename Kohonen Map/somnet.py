# File: somnet.py
# Copyright 2009 Justin Sonntag. All rights reserved.

import numpy as np
import random
from Tkinter import *

FEEDBACK = False
XSIZE = 20
YSIZE = 20
CELL_SIZE = 10

EXAMPLE1 = np.array([[255,0,0],[0,255,0],[0,0,255]])
EXAMPLE2 = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],
                     [0,255,255],[255,255,255],[0,0,0]])
EXAMPLE3 = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],
                     [0,255,255],[0,0,0]])

class somnet:
    "An implementation of a Self-Ordering Map"

    def __init__(self, data, feedback = False):
        "Creates a new 2D Self-Ordering Map of the same size as data."
        self.net = data
        self.feedback = feedback

    def train(self, data, time):
        "Train the map to classify data"
        endtime = time
        for c in range(time):
            currdata = random.choice(data)
            location = self.activate(currdata)
            for x in range(self.net.shape[0]):
                for y in range(self.net.shape[1]):
                    self.learn(currdata, (x,y), location, c, endtime)
            if self.feedback and c % endtime/10 == 0:
                print self.net
                #createpicture(self)

    def neighbour(self, array, currtime, endtime):
        "A gaussian function to act as the neighbourhood function"
        #return np.exp(-((starttime * 1.0) / currtime) * np.dot(array, array))
        return np.exp(-((currtime * 1.0)/endtime + .1) * np.dot(array, array))

    def timerestraint(self, currtime, endtime):
        return ((endtime - currtime) * 1.0) / endtime

    def learn(self, target, location, neighbour, currtime, endtime):
        self.net[location[0], location[1]] += (
            self.neighbour(location - neighbour, currtime, endtime) *
            self.timerestraint(currtime, endtime) *
            (target - self.net[location[0], location[1]]))

##    def activate(self, array):
##        neuron = None
##        smallest = np.inf
##        for x in range(self.net.shape[0]):
##            for y in range(self.net.shape[1]):
##                curritem = array - self.net[x,y]
##                currdist = np.dot(curritem, curritem)
##                if currdist < smallest:
##                    smallest = currdist
##                    neuron = np.array([x,y])
##        return neuron
    def activate(self, array):
        smallest = np.inf
        for x in range(self.net.shape[0]):
            for y in range(self.net.shape[1]):
                curritem = array - self.net[x,y]
                currdist = np.dot(curritem, curritem)
                if currdist < smallest:
                    smallest = currdist
        possibles = []
        for x in range(self.net.shape[0]):
            for y in range(self.net.shape[1]):
                curritem = array - self.net[x,y]
                if np.dot(curritem, curritem) == smallest:
                    possibles.append(np.array([x,y]))
        return random.choice(possibles)

def getrandomnet(xsize, ysize):
    data = np.empty((xsize, ysize, 3), dtype = int)
    for x in range(xsize):
        for y in range(ysize):
            for z in range(3):
                data[x,y,z] = random.choice(range(256))
    network = somnet(data, FEEDBACK)
    return network

class rectangle:
    def __init__(self, master, x1, y1, x2, y2, fill = "white"):
        self.canvas = master
        self.id = self.canvas.create_rectangle(x1, y1, x2, y2, fill = fill,
                                               width = "0")
        self.config = {"fill": fill, "width": "0" }

    def setfill(self, color):
        self.config["fill"] = color
        self.canvas.itemconfig(self.id, self.config)

def createpicture(network):
    root = Tk()
    field = Canvas(root, width = network.net.shape[0] * CELL_SIZE,
                   height = network.net.shape[1] * CELL_SIZE)
    field.pack()
    for x in range(network.net.shape[0]):
        for y in range(network.net.shape[1]):
            rectangle(field, x * CELL_SIZE, y * CELL_SIZE, (x+1) * CELL_SIZE,
                      (y+1) * CELL_SIZE, getcolor(network.net[x,y]))

def getcolor(array):
#    return "#%02x%02x%02x" % tuple(array / 5.0 * 255)
    return "#%02x%02x%02x" % tuple(array)
