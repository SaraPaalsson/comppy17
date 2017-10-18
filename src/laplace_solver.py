class SimClass:
	"""
	Simulation class. Contains all variables and needed parameters 
	to solve Laplace's equation on a domain.
	"""

	def readInput(self):
		f = open(self.str_name,'r')	
		data = f.readline().strip().split(' ')
		f.close()
		print(data)
		self.nbr_interf = int(data[0]) #nbr interface points
		self.nbr_dom = int(data[1]) #nbr domain points
		self.shape = str(data[2]) #shape of interface
		nbr_src = int(data[3]) #nbr sources for RHS
		self.src = []
		for i in range(nbr_src):
			x = data[4+i]
			x2 = x.split('+')
			xr = int(x2[0])
			xi = x2[1]
			if xi[0] is '-':
				xi = -int(xi[1:-1])
			else:
				xi = int(xi[0:-1])
			self.src.append(complex(xr,xi))

	def __init__(self,str):
		self.str_name = str + '.dat'
		self.readInput()


def test_setup_SimClass():
	sc = SimClass('sim_test')
	assert sc.nbr_interf == 1

def test_create():
		sc = SimClass('sim_test')


if __name__ == '__main__':
	print("Running main program.")

	print("Defining simulation...")
	sc = SimClass('sim1')
	print(sc.src)

#	print("Setting up domain...")


