import SimClass as simc
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
	print("Running main program.")

	print("Defining simulation...")
	sc = simc.SimClass('sim1')

	print("Setting up domain...")
	sc.setUp()

#	# PLOT
#	plt.plot(sc.zpanels.real,sc.zpanels.imag)
#	plt.show()

