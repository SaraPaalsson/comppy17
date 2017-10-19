import SimClass as simc
import numpy as np
import matplotlib.pyplot as plt 

def setUpSim(inp = ''):
	print("Defining simulation...")
	sc = simc.SimClass(inp)
	print("Setting up domain...")
	sc.setUp()
	return sc


if __name__ == '__main__':
	print("Running main program.")

	#inputn = 'sim1'
	sc = setUpSim()

#	# PLOT
#	plt.plot(sc.zpanels.real,sc.zpanels.imag)
#	plt.show()

