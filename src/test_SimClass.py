import SimClass as simc
import numpy as np
import math

def test_readInput():
	"""
	Test initialization of simulation class and reading of input.
	"""
	sc = simc.SimClass()
	assert sc.nbr_panels == 2
	assert sc.nbr_dom == 1
	assert sc.shape == 'circle'
	assert sc.radius == 2
	assert sc.src[0] == 3+3j 
	assert sc.src[1] == -2.5-2.5j
	assert sc.fillLevel == 'superlow'

def test_createInterface():
	""" 
	Test creation of interface.
	"""
	sc = simc.SimClass()
	sc.createInterface()
	assert sc.tpanels[0] == 0
	assert sc.tpanels[2] == 2*math.pi
	assert np.abs(sc.zpanels[0]-2) < 10**(-13)
	assert np.abs(sc.zpanels[2]-2) < 10**(-13)

def test_gaussLeg():
	"""
	Test Gauss-Legendre quadrature nodes and weights
	"""
	sc = simc.SimClass()
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

def test_fillDomain():
	"""
	Test fillDomain routine.
	"""
	sc = simc.SimClass()
	sc.fillDomain()
	tmp = np.shape(sc.zDom)
	assert tmp[0] == 10 
	assert tmp[1] == 14

def test_rhsf():
	"""
	Test computation of rhs function.
	"""
	sc = simc.SimClass()
	sc.createInterface()
	r = sc.rhsf(0)
	assert np.abs(r - np.real((-1/(3+3j) - 1/(-2.5-2.5j)))) < 10**(-13)
	r = sc.rhsf()
	assert max(np.abs(r-np.real(1/(sc.zDrops-(3+3j)) + 1/(sc.zDrops-(-2.5-2.5j))))) < 10**(-13)

def test_setUp():
	"""
	Test setUp() function.
	"""
	sc = simc.SimClass()
	sc.setUp()
	assert sc.tpanels[0] == 0
	assert sc.tpanels[2] == 2*math.pi
	assert np.abs(sc.zpanels[0]-2) < 10**(-13)
	assert np.abs(sc.zpanels[2]-2) < 10**(-13)
	tmp = np.shape(sc.zDom)
	assert tmp[0] == 10 
	assert tmp[1] == 14
	assert max(np.abs(sc.RHS-np.real(1/(sc.zDrops-(3+3j)) + 1/(sc.zDrops-(-2.5-2.5j))))) < 10**(-13)








