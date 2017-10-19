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
	assert np.abs(sc.zpanels[0]-2) < 10**(-13)
	assert np.abs(sc.zpanels[1]-2) < 10**(-13)

def test_gaussLeg():
	"""
	Test Gauss-Legendre quadrature nodes and weights
	"""
	sc = SimClass()
	n,w  = sc.gaussLeg(16,-1,1)
	w_corr = np.array([0.027152459411754, 0.062253523938648, 0.095158511682493, \
	 0.124628971255534, 0.149595988816577, 0.169156519395003, 0.182603415044924, \
	 0.189450610455069, 0.189450610455069, 0.182603415044924, 0.169156519395003, \
	 0.149595988816577, 0.124628971255534, 0.095158511682493, 0.062253523938648, \
	 0.027152459411754])
	n_corr =  np.array([-0.989400934991650, -0.944575023073233, -0.865631202387832,\
	 -0.755404408355003, -0.617876244402644, -0.458016777657227, -0.281603550779259,\
	  -0.095012509837637, 0.095012509837638, 0.281603550779259, 0.458016777657228, \
	  0.617876244402644, 0.755404408355003, 0.865631202387831, 0.944575023073233, \
	  0.989400934991650])
	assert np.abs(sum(w)-2) < 10**(-13)
	assert max(np.abs(w-w_corr)) < 10**(-13)
	assert max(np.abs(n-n_corr)) < 10**(-13)

if __name__ == "__main__":
	print('Simulation class')