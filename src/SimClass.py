import numpy as np
import math
import scipy.linalg as linalg

class SimClass(object):
	"""
	Simulation class. Contains all variables and needed parameters 
	to solve Laplace's equation on a domain.
	"""
	def __init__(self,str=""):
		self.str_name = str + '.dat'
		self.readInput()

	def readInput(self):
		if self.str_name == ".dat": #If no inputfile defined. For testing.
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

	def createInterface(self):
		# Create panels
		self.tpanels = np.linspace(0,2*math.pi,self.nbr_panels+1)
		self.zpanels = self.radius*np.cos(self.tpanels) + self.radius*1j*np.sin(self.tpanels)

		# Create empty arrays for interface points and weights
		self.tDrops = np.zeros(self.nbr_panels*16)
		self.wDrops = np.zeros(self.nbr_panels*16)
		for i in range(self.nbr_panels): #Go through all panels!
			n,w = self.gaussLeg(16,self.tpanels[i],self.tpanels[i+1])
			self.tDrops[i*16:(i+1)*16] = n
			self.wDrops[i*16:(i+1)*16] = w
		self.zDrops = (self.radius*np.cos(self.tDrops) + 
			self.radius*1j*np.sin(self.tDrops))
		self.zpDrops = (-self.radius*np.sin(self.tDrops) +
			self.radius*1j*np.cos(self.tDrops))
		self.zppDrops = -self.zDrops

	def gaussLeg(self,n,t1,t2):
		"""
		Create Gauss-Legendre nodes and weights of order n, on interval [t1,t2]. 
		As done in Trefethen.
		"""
		n_vec = np.linspace(1,15,15)
		beta = 0.5*(1-(2*n_vec)**(-2))**(-1/2)
		T = np.diag(beta,-1) + np.diag(beta,1)
		D,V = linalg.eig(T)
		nodes = np.real((t1*(1-D)+t2*(1+D))/2) #Remap to [t1,t2]
		weights = 2*(V[0]**2).T
		weights = (t2-t1)/2*weights
		idx = np.argsort(nodes)
		nodes = np.array(nodes)[idx]
		weights = np.array(weights)[idx]
		return nodes, weights

if __name__ == "__main__":
	print('Simulation class')
