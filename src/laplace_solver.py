import SimClass as simc
import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse.linalg as spla
import math

def setUpSim(inp = ''):
	print("Defining simulation...")
	sc = simc.SimClass(inp)
	print("Setting up domain...")
	sc.setUp()
	return sc

def compDensity(sc):
	"""
	Compute complex density for double layer formulation.
	"""
	N = sc.nbr_panels*16

	# OK
	#RHS, wdrops, zDrops, zpDrops, zppDrops, d

	# Compute limit points
	d = sc.wDrops*np.imag(sc.zppDrops/(2*sc.zpDrops))/math.pi

	# Set up matrix and rhs
	A = np.eye(N)
	zDrops = np.copy(sc.zDrops)
	for i in range(N):
		zDrops[i] = zDrops[i]*2
		tmp = sc.wDrops*np.imag(sc.zpDrops/(zDrops-sc.zDrops[i]))/math.pi
		tmp[i] = 0
		A[i,:] += tmp
		A[i,i] += d[i]
		zDrops[i] = sc.zDrops[i]
	
	b = 2*sc.RHS

	# Compute mu through gmres
	mu,conv = spla.gmres(A,b) 

	if conv == 0:
		print("GMRES converged succesfully.")
	else:
		print("GMRES did not converge.")
	return mu

def compSolStandard(sc,mu,ztarg=-100):
	"""
	Compute solution using standard quadrature rules.
	"""
	if ztarg == -100:
		ztarg = sc.zDom

	N = np.size(ztarg) #Number of target points
	u = np.copy(ztarg)

	ztmp = ztarg.reshape(N)
	utmp = u.reshape(N)
	for i in range(N):
		tmp = sc.wDrops*mu*np.imag(sc.zpDrops/(sc.zDrops-ztmp[i]))
		utmp[i] = sum(0.5*tmp/math.pi)
	return u

def compSolSpecial(sc,mu,u):
	"""
	Correct solution u using special quadrature. 
	"""
	return None

def compSolCorrect(sc,ztarg=-100):
	"""
	Compute correct solution using RHS analytic.
	"""
	if ztarg == -100:
		ztarg = sc.zDom
		ztarg = ztarg.reshape(np.size(ztarg))
	ucorrect = sc.rhsf(ztarg)
	return ucorrect

def compError(sc,u,uspecial,ucorrect):
	"""
	Compute errors.
	"""
	u2 = u.reshape(np.size(u))
	uc2 = ucorrect.reshape(np.size(ucorrect))
	est = np.abs(u2-uc2)

	return est, None

if __name__ == '__main__':
	print("Running main program.")

	inputn = 'sim1'
	#inputn = ''
	sc = setUpSim(inputn)

	print("Computing density...")
	mu = compDensity(sc)

	print("Computing correct solution...")
	ucorrect = compSolCorrect(sc)

	print("Computing u with standard quadratre...")
	u = compSolStandard(sc,mu)
	
	print("Computing errors")
	est, esp = compError(sc,u,'',ucorrect)
	print("Error using standard quadrature is {}".format(max(est)))
	#print(est)
	#est2 = est.reshape(np.shape(u))

	z = sc.zDom
	z = z.reshape(np.size(z))

	fig, ax = plt.subplots()
	cont = ax.tricontourf(np.real(z),np.imag(z),np.log10(est))
	cb = fig.colorbar(cont)
	cb.set_label('10log error')
	line, = ax.plot(np.real(sc.zDrops), np.imag(sc.zDrops), 'k.-')
	ax.axis('equal')
	ax.set_title('Error with standard quadrature')
	ax.set_xlabel('x')
	ax.set_ylabel('y')

	plt.show()

	#print(u.reshape(np.size(u)))

	#tmp = sc.zDom
	#print(tmp.reshape(np.size(tmp)))

#	# PLOT
#	plt.plot(sc.zpanels.real,sc.zpanels.imag)
#	plt.show()

