from src import laplace_solver as lapls
import math
import numpy as np

def test_setUpSim():
	sc = lapls.setUpSim()
	assert sc.tpanels[0] == 0
	assert sc.tpanels[1] == math.pi
	assert np.abs(sc.zpanels[0]-2) < 1e-13
	assert np.abs(sc.zpanels[2]-2) < 1e-13
	tmp = np.shape(sc.zDom)
	assert tmp[0] == 10 
	assert tmp[1] == 14
	assert max(np.abs(sc.RHS-np.real(1/(sc.zDrops-(3+3j)) + 1/(sc.zDrops-(-2.5-2.5j))))) < 1e-13

def test_compDenisty():
	sc = lapls.setUpSim()
	mu = lapls.compDensity(sc)
	mu_test = np.array([0.100066359606831, 0.071545644107794, 0.011300115712400, -0.097687110933043, -0.267698478312211, -0.447901998179010, -0.520684110484826, -0.480033494442193, -0.409789531064752, -0.352050592733239, -0.311277756165332, -0.281248613350712, -0.254802207222953, -0.227315520617973, -0.199676427248985, -0.179131868019445, -0.167788242379403, -0.139541229603738, -0.071451471961443, 0.079694597163585, 0.379852420019739, 0.734601886024344, 0.796884222404543, 0.641368572406435, 0.489061566293341, 0.383510970741658, 0.311633966954139, 0.258182250686599, 0.213023984397086, 0.171720849353246, 0.136010951922997, 0.112327845842463])
	assert max(np.abs(mu - mu_test)) < 1e-13

def test_compSolStandard():
	sc = lapls.setUpSim()
	mu_ones = np.ones(sc.nbr_panels*16)
	zinside = np.array([0])
	zoutside = np.array([10])
	uinside1 = lapls.compSolStandard(sc,mu_ones,zinside)
	uoutside1 = lapls.compSolStandard(sc,mu_ones,zoutside)
	assert np.abs(uinside1 - 1) < 1e-13
	assert np.abs(uoutside1 - 0) < 1e-13

def test_compSolCorrect():
	sc = lapls.setUpSim()
	uc = lapls.compSolCorrect(sc)
	assert np.abs(uc[0] - 0.033333333333333) < 1e-13
	assert np.abs(uc[-1] - 0.148563895918862) < 1e-13

def test_compError():
	sc = lapls.setUpSim()
	u = np.ones((3,2)) 
	us = np.ones((3,2))
	uc = np.ones((3,2))*2
	uc[0] = 1
	est, esp = lapls.compError(sc,u,us,uc)
	assert max(est) == 1
	assert max(esp) == 1
	assert est[0] == 1e-17
	assert esp[0] == 1e-17