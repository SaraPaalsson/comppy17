import numpy as np
import math
import scipy.linalg as linalg
import matplotlib.pyplot as plt

class SimClass(object):
	"""
	Simulation class. Contains all variables and needed parameters 
	to solve Laplace's equation on a domain.
	"""
	def __init__(self,data=''):
		self.data = data
		self.readInput()

	def readInput(self):
		"""
		Read input data and assign parameters
		"""
		if self.data == '': #If no input data is defined
			self.data = [2,'circle',2,2,3+3j,-2.5-2.5j, 'superlow']
	
		self.nbr_panels = self.data[0] #nbr interface points
		self.shape = self.data[1] #shape of interface
		self.radius = self.data[2]
		nbr_src = self.data[3] #nbr sources for RHS
		self.src = [] #sources for RHS
		for i in range(nbr_src):
			x = self.data[4+i]
			self.src.append(x)
		self.fillLevel = (self.data[-1])

	def setUp(self):
		"""
		Set up simulation.
		"""
		self.createInterface()
		self.fillDomain()
		self.RHS = self.rhsf()

	def rhsf(self,x=-100):
		"""
		Compute RHS given by sources.
		"""
		if np.size(x) == 1: 
			if x == -100:
				x = self.zDrops
				f = np.zeros(np.shape(x))
				for j in range(np.size(f)):
					for i in range(len(self.src)):
						f[j] += np.real(1/(x[j]-self.src[i]))
			else:
				f = 0
				for i in range(len(self.src)):
						f += np.real(1/(x-self.src[i]))
		else:
			f = np.zeros(np.shape(x))
			for j in range(np.size(f)):
				for i in range(len(self.src)):
					f[j] += np.real(1/(x[j]-self.src[i]))
		return f

	def fillDomain(self):
		"""
		Fill domain (interior) with computational points for evaluating solution.
		"""
		if self.fillLevel == "superlow":
			nbrR = 10
			nbrT = 10
		if self.fillLevel == "low":
			nbrR = 20
			nbrT = 20
		if self.fillLevel == 'medium':
			nbrR = 50
			nbrT = 50
		R1 = 0.4 #where to go from sparse to dense disc. in domain
		r1 = np.linspace(0,R1,5)
		r2 = np.linspace(R1,0.999,nbrR); 
		r2 = r2[1:]
		r = np.append(r1, r2) #radial discretisation
		t = np.linspace(0,2*math.pi,nbrT+1)
		#t = t[0:-1]
		R,T = np.meshgrid(r,t)

		self.zDom = np.zeros((np.size(t),np.size(r)))
		zD = self.zDom
		zD = zD.reshape(np.size(zD))
		RD = R.reshape(np.size(R))
		TD = T.reshape(np.size(T))

		if self.shape == 'circle':
			zD = RD*(self.radius*np.cos(TD) + self.radius*1j*np.sin(TD))
		if self.shape == 'starfish':
			zD = RD*((1+0.3*np.cos(5*TD))*np.exp(1j*TD))
		self.zDom = zD.reshape(np.shape(self.zDom))

	def createInterface(self):
		"""
		Create interface discretization.
		"""
		# Create panels
		self.tpanels = np.linspace(0,2*math.pi,self.nbr_panels+1)
		if self.shape == 'circle':
			self.zpanels = self.radius*np.cos(self.tpanels) + self.radius*1j*np.sin(self.tpanels)
		if self.shape == 'starfish':
			self.zpanels = (1+0.3*np.cos(5*self.tpanels))*np.exp(1j*self.tpanels)

		# Create empty arrays for interface points and weights
		self.tDrops = np.zeros(self.nbr_panels*16)
		self.wDrops = np.zeros(self.nbr_panels*16)
		for i in range(self.nbr_panels): #Go through all panels!
			n,w = self.gaussLeg(16,self.tpanels[i],self.tpanels[i+1])
			self.tDrops[i*16:(i+1)*16] = n
			self.wDrops[i*16:(i+1)*16] = w
	
		if self.shape == 'circle':
			self.zDrops = (self.radius*np.cos(self.tDrops) + 
				self.radius*1j*np.sin(self.tDrops))
			self.zpDrops = (-self.radius*np.sin(self.tDrops) +
				self.radius*1j*np.cos(self.tDrops))
			self.zppDrops = -self.zDrops
		if self.shape == 'starfish':
			self.zDrops = (1+0.3*np.cos(5*self.tDrops))*np.exp(1j*self.tDrops)
			self.zpDrops = (-1.5*np.sin(5*self.tDrops)+1j*(1+0.3*np.cos(5*self.tDrops)))*np.exp(1j*self.tDrops)
			self.zppDrops = np.exp(1j*self.tDrops)*(-1-7.8*np.cos(5*self.tDrops)-3j*np.sin(5*self.tDrops))


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

	def plotInterface(self,ax=''):
		if ax == '':
			fig,ax = plt.subplots()
		line = ax.plot(np.real(self.zDrops),np.imag(self.zDrops),'k-',linewidth=2.0)
		ax.axis('equal')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		#plt.show()
		return fig,ax

	def plotPanels(self,ax=''):
		if ax == '':
			fig,ax = plt.subplots()
		for pp in self.zpanels:
			pl = ax.plot(np.real(pp),np.imag(pp),'bd',markersize=8)

	def plotSources(self,ax=''):
		if ax == '':
			fig,ax = plt.subplots()
		for zi in self.src:
			pz = ax.plot(np.real(zi),np.imag(zi),'r*',markersize=8)

	def plotRHS(self,ax=''):
		if ax == '':
			fig,ax = plt.subplots()
		line1, = ax.plot(self.tDrops,np.real(self.RHS),'-',label='$\Re(f)$')
		line2, = ax.plot(self.tDrops,np.imag(self.RHS),'--',label='$\Im(f)$')
		ax.legend()
		ax.set_xlabel('t')
		return fig,ax


if __name__ == "__main__":
	print('Simulation class')
	sc = SimClass()
	sc.setUp()

