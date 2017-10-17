def mymax(x, y):
	"""
	Returns the max of two numbers.

	Input parameters: x, y (numbers).
	"""
	if x > y:
		return x
	else:
		return y

def listmax(l):
	"""
	Returns max number from a list of positive numbers.
	"""
	lmax = 0
	for  i in l:
		if i > lmax:
			lmax = i
	return lmax

if __name__ == "__main__":
	import sys
	mylist = []
	for arg in sys.argv[1:]:
		mylist.append(int(arg))
	print(listmax(mylist))



