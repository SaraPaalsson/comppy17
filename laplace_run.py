import matplotlib.pyplot as plt 
import numpy as np
from src import SimClass as simc
from src import laplace_solver as ls


if __name__ == '__main__':
	print("Running main program.")

	inputdata = [10, 'circle', 2, 2, 3+3j, -3-3j, 'medium']
	#inputdata = ''
	sc = ls.setUpSim(inputdata)

	mu = ls.compDensity(sc)

	ucorrect = ls.compSolCorrect(sc)

	u = ls.compSolStandard(sc,mu)

	usc = ls.compSolSpecial(sc,mu,u)

	est, esp = ls.compError(sc,u,usc,ucorrect)
	print("Error using standard quadrature is {}".format(max(est)))
	print("Error using special quadrature is {}".format(max(esp)))

	z = sc.zDom
	z = z.reshape(np.size(z))

	est2 = est;
	est2 = est2.reshape(np.shape(sc.zDom))
	esp2 = esp
	esp2 = esp2.reshape(np.shape(sc.zDom))

	#fig, ax = plt.subplots(nrows=1,ncols=2)
	#ax1 = ax[0]
	fig, ax1 = plt.subplots()
	cont = ax1.tricontourf(np.real(z),np.imag(z),np.log10(est),vmin=-12,vmax=0)
	cb = fig.colorbar(cont)
	cb.set_label('10log error')
	line, = ax1.plot(np.real(sc.zDrops), np.imag(sc.zDrops), 'k.-')
	ax1.axis('square')
	ax1.set_title('Standard quadrature, Np = {}'.format(sc.nbr_panels))
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')

	fig, ax2 = plt.subplots()
	#ax2 = ax[1]
	cont = ax2.tricontourf(np.real(z),np.imag(z),np.log10(esp),vmin=-12,vmax=0)
	cb = fig.colorbar(cont)
	cb.set_label('10log error')
	line, = ax2.plot(np.real(sc.zDrops), np.imag(sc.zDrops), 'k.-')
	ax2.axis('square')
	ax2.set_title('Special quadrature')
	ax2.set_xlabel('x')
	ax2.set_ylabel('y')

	plt.show()
