import numpy as np
import math

class SimClass(object):
	"""
	Simulation class. Contains all variables and needed parameters 
	to solve Laplace's equation on a domain.
	"""
	def __init__(self,str=""):
		self.str_name = str + '.dat'
		self.readInput()

	def readInput(self):
		if self.str_name == ".dat":
			data = ['1','1','circle','2','2','1+1j','-2+-2j']
		else:
			f = open(self.str_name,'r')	
			data = f.readline().strip().split(' ')
			f.close()	
		self.nbr_panels = int(data[0]) #nbr interface points
		self.nbr_dom = int(data[1]) #nbr domain points
		self.shape = str(data[2]) #shape of interface
		self.radius = int(data[3])
		nbr_src = int(data[4]) #nbr sources for RHS
		self.src = [] #sources for RHS
		for i in range(nbr_src):
			x = data[5+i]
			x2 = x.split('+')
			xr = int(x2[0])
			xi = x2[1]
			if xi[0] is '-':
				xi = -int(xi[1:-1])
			else:
				xi = int(xi[0:-1])
			self.src.append(complex(xr,xi))

	def setUp(self):
		self.createInterface()
		print("Hello")

	def createInterface(self):
		# Create panels
		self.tpanels = np.linspace(0,2*math.pi,self.nbr_panels+1)
		self.zpanels = self.radius*np.cos(self.tpanels) + self.radius*1j*np.sin(self.tpanels)

		# Create empty arrays for interface points and weights
		self.zDrops = np.zeros(self.nbr_panels*16)
		self.wDrops = np.zeros(self.nbr_panels*16)
		for i in range(self.nbr_panels): #Go through all panels!
			tmp = self.gaussleg(16,self.tpanels[i],self.tpanels[i+1])


		#self.tpar = self.tpar[:-1] #Remove 2*pi (periodic)
		#if self.shape == 'circle':
		#	self.zDrops = r*np.cos(self.tpar) + r*1j*np.sin(self.tpar)
		#	self.zpDrops = -r*np.sin(self.tpar) + r*1j*np.cos(self.tpar)
		#	self.zppDrops = -r*np.cos(self.tpar) - r*1j*np.sin(self.tpar)

	def gaussleg(slef,n,t1,t2):
		print("Gaussleg")


def test_readInput():
	"""
	Test initialization of simulation class and reading of input.
	"""
	sc = SimClass()
	assert sc.nbr_panels == 1
	assert sc.nbr_dom == 1
	assert sc.shape == 'circle'
	assert sc.radius == 2
	assert sc.src[0] == 1+1j 
	assert sc.src[1] == -2-2j

def test_createInterface():
	""" 
	Test creation of interface.
	"""
	sc = SimClass()
	sc.createInterface()
	assert sc.tpanels[0] == 0
	assert sc.tpanels[1] == 2*math.pi


if __name__ == "__main__":
	print('Simulation class')