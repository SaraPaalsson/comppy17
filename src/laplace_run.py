import matplotlib.pyplot as plt 
import numpy as np
import SimClass as simc
import laplace_solver as ls


if __name__ == '__main__':
	print("Running main program.")

	inputn = 'sim1'
	#inputn = ''
	sc = ls.setUpSim(inputn)

	print("Computing density...")
	mu = ls.compDensity(sc)

	print("Computing correct solution...")
	ucorrect = ls.compSolCorrect(sc)

	print("Computing u with standard quadratre...")
	u = ls.compSolStandard(sc,mu)
	
	print("Computing errors")
	est, esp = ls.compError(sc,u,'',ucorrect)
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
	ax.set_title('Error using standard quadrature, Np = {}'.format(sc.nbr_panels))
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	plt.show()
