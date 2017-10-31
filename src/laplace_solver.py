import SimClass as simc
import numpy as np
import scipy.sparse.linalg as spla
import math
import matplotlib.pyplot as plt

W16 = np.array([0.027152459411754, 0.062253523938648, 0.095158511682493, \
	0.124628971255534, 0.149595988816577, 0.169156519395003, 0.182603415044924, \
	0.189450610455069, 0.189450610455069, 0.182603415044924, 0.169156519395003, \
	0.149595988816577, 0.124628971255534, 0.095158511682493, 0.062253523938648, \
	0.027152459411754])

W32 = np.array([0.007018610009470, 0.016274394730906, 0.025392065309262, \
	0.034273862913021, 0.042835898022227, 0.050998059262376, 0.058684093478536, \
	0.065822222776362, 0.072345794108849, 0.078193895787070, 0.083311924226947, \
	0.087652093004404, 0.091173878695764, 0.093844399080805, 0.095638720079275, \
	0.096540088514728, 0.096540088514728, 0.095638720079275, 0.093844399080805, \
	0.091173878695764, 0.087652093004404, 0.083311924226947, 0.078193895787070, \
	0.072345794108849, 0.065822222776362, 0.058684093478536, 0.050998059262376, \
	0.042835898022227, 0.034273862913021, 0.025392065309262, 0.016274394730906, \
	0.007018610009470])

IP1 = np.array([0.708233680592340, 0.417460345661405, 0.120130621067993, -0.023738421449022, \
	-0.024711509929699, 0.008767765249452, 0.010348492268217, -0.004882937574295, -0.005817138599704, \
	0.003371610049761, 0.003906389536232, -0.002688491450644, -0.002982620437980, 0.002396613365728, \
	0.002527899637498, -0.002352088012027, -0.353398976603946, 0.128914102924785, 0.490438143959320, \
	0.447688855448316, 0.159581518743026, -0.043199604388389, -0.045054276136530, 0.019843278050097, \
	0.022653307118776, -0.012766166617303, -0.014509184948602, 0.009852414829341, 0.010828396616511, \
	-0.008645914067315, -0.009083716164910, 0.008436024529321, 0.254788814089210, -0.079438651130554, \
	-0.180021675391748, 0.112706994698110, 0.452990321322730, 0.457103093490146, 0.172045630255837, \
	-0.057268690116309, -0.057207007970563, 0.029864275385776, 0.032348060459762, -0.021279595795857, \
	-0.022894435002697, 0.018024807626862, 0.018775629092083, -0.017365908001573, -0.192642764476253, \
	0.057539598371389, 0.118550200216509, -0.060558741914225, -0.137550413480163, 0.110511198044240, \
	0.438506321611140, 0.462013534571532, 0.176750576258897, -0.070000077178178, -0.066362764802900, \
	0.040455161093420, 0.041534393253997, -0.031758507171589, -0.032516100034128, 0.029834063376996, \
	0.143303354000051, -0.042055381151416, -0.083623389175777, 0.040136114296907, 0.081653505573269, \
	-0.052756328748486, -0.114499477585301, 0.111212881525809, 0.430368039226389, 0.466243258020893, \
	0.179333130796333, -0.083730103755132, -0.075715752328325, 0.054025580681878, 0.053262695336761, \
	-0.048049159001486, -0.099719237840432, 0.029015527255332, 0.056745671351501, -0.026507281530925, \
	-0.051684172094834, 0.031210872338250, 0.060395080215009, -0.046871396211720, -0.096541871337564, \
	0.111392998543353, 0.422933391767461, 0.472656340478166, 0.183945672706357, -0.102439013196790, \
	-0.090682596162259, 0.078322264361212, 0.058956626916838, -0.017080869741123, -0.033131760949494, \
	0.015276769255016, 0.029215868945318, -0.017147768479290, -0.031798751860556, 0.023087392278153, \
	0.042463097189794, -0.039147738182975, -0.077430953866654, 0.106967836431448, 0.409850258260916, \
	0.488757091888883, 0.202162218471372, -0.145558364176374, -0.019521496677808, 0.005645327810182, \
	0.010912188921697, -0.005004288804177, -0.009495119079648, 0.005510772494078, 0.010056981232185, \
	-0.007134062523268, -0.012669001886025, 0.011041839978672, 0.019781931058369, -0.022233561830741, \
	-0.044565913068780, 0.079639340872343, 0.355553969823584, 0.596733166923931])

IP2 = np.array([0.713862126485003, 0.415861464999546, 0.117139053388567, -0.022430941452515, \
	-0.022386727521234, 0.007526833242539, 0.008309785375195, -0.003613499292735, -0.003898339148532, \
	0.002002775905536, 0.002001361056109, -0.001144934539207, -0.001000441823818, 0.000579622749829, \
	0.000369122977751, -0.000114841089527, -0.373111737631011, 0.134514697868894, 0.500919671211399, \
	0.443106180943123, 0.151429254240996, -0.038845347375560, -0.037895234846542, 0.015381407522509, \
	0.015901484842618, -0.007943124788689, -0.007786257681468, 0.004394915660383, 0.003804467366061, \
	-0.002190252675426, -0.001389346807558, 0.000431437040723, 0.293533407753495, -0.090449199150776, \
	-0.200637543016583, 0.121726728257118, 0.469050569386606, 0.448514982681450, 0.157904965796408, \
	-0.048439925399146, -0.043818636110253, 0.020276192880273, 0.018942511378929, -0.010357973256273, \
	-0.008777345506291, 0.004982616916114, 0.003133611585860, -0.000969126893520, -0.254321612548312, \
	0.075074608907870, 0.151405998292083, -0.074948908347047, -0.163209724781879, 0.124257459362683, \
	0.461191598958519, 0.447810530211124, 0.155140021647846, -0.054461091185942, -0.044531484146797, \
	0.022565176432982, 0.018247128138763, -0.010060054357776, -0.006218741515135, 0.001907870729615, \
	0.231294304516010, -0.067085064623796, -0.130570952180064, 0.060729794130518, 0.118450523305909, \
	-0.072521831781564, -0.147226860414845, 0.131787043059898, 0.461828826830738, 0.443484454902563, \
	0.147123228142680, -0.057098466541441, -0.040667821628640, 0.020922699023661, 0.012453895329684, \
	-0.003756646627295, -0.217123906984604, 0.062438843010710, 0.119528551263843, -0.054106792084372, \
	-0.101143929931585, 0.057878893205722, 0.104762346987468, -0.074928255602672, -0.139758055668850, \
	0.142936730014770, 0.468072149819296, 0.434818901723193, 0.133282875813318, -0.053518478918945, \
	-0.028603957732716, 0.008260758005463, 0.208787542905641, -0.059782988522232, -0.113508058889059, \
	0.050717913029473, 0.092991730208347, -0.051720793838675, -0.089713332858577, 0.060028276446192, \
	0.099980675207689, -0.081702601145486, -0.139379433799503, 0.160051371016460, 0.483006808606042, \
	0.415312218104070, 0.103715923253338, -0.024969801606650, -0.204900209448852, 0.058561762926483, \
	0.110802967056481, -0.049241305347759, -0.089574268926796, 0.049263741070694, 0.084095332698592, \
	-0.054976265993162, -0.088410558594981, 0.068301146401525, 0.105538302999532, -0.098599012554463, \
	-0.155663999454775, 0.200570302178176, 0.540640169981230, 0.303399903673815])

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
	uspecial = np.copy(u)
	N = np.size(u)
	us = uspecial.reshape(N)
	z = sc.zDom
	z = z.reshape(np.size(z))

	for i in range(N): #Go through all domain points
		for k in range(sc.nbr_panels): #Go through all panels
			# Calculate mid points and lengths of panel
			mid2 = (sc.zpanels[k+1]+sc.zpanels[k])/2
			len2 = sc.zpanels[k+1]-sc.zpanels[k]

			#if 1:
			if np.abs(z[i]-mid2) < np.abs(len2): #Check if z[i] too close to panel k
				nz = 2*(z[i]-mid2)/len2; #remap z[i] to [-1,1] int.
				lg1 = np.log(1-nz)
				lg2 = np.log(-1-nz)

				j = np.arange(16)
				indj = k*16 + j
				tz = sc.zDrops[indj] #take out points on interface panel k
				nzpan = 2*(tz-mid2)/len2 #remap panel nodes to [-1,1]

				#Check if the point nz is between the panel and the real axis
				if np.real(nz) > -1 and np.real(nz) < 1: 
					if np.imag(nz) > 0:
						furthercheck = 0 
						for ij in range(16):
							if np.imag(nzpan[ij]) > np.imag(nz):
								furthercheck = 1               
								break
						if furthercheck: 
							tmpT = np.real(nzpan)
							tmpb = np.imag(nzpan)
							p = vandernewtonT(tmpT,tmpb,16)
							interpk = np.arange(0,16)
							test = sum(p*np.real(nz)**interpk)
							if test > np.imag(nz): #correct value of integral
								lg1 -= 1j*math.pi
								lg2 =+ 1j*math.pi

					else:
						if np.imag(nz) < 0: #below the real axis
							furthercheck = 0 
							for ij in range(16):
								if np.imag(nzpan[ij]) < np.imag(nz):
									furthercheck = 1
									break
							if furthercheck:
								tmpT = np.real(nzpan)
								tmpb = np.imag(nzpan)
								p = vandernewtonT(tmpT,tmpb,16)
								interpk = np.arange(0,16)
								test = sum(p*np.real(nz)**interpk)
							
								if test < np.imag(nz):
									lg1 += 1j*math.pi
									lg2 -= 1j*math.pi

				p32 = np.zeros(32,dtype=np.complex_)
				p32[0] = lg1-lg2
				tzp = sc.zpDrops[indj]
				tmu = mu[indj]
				tW = sc.wDrops[indj]

				#Compute panel contribution to u with standard quadrature
				oldsum = 1/(2*math.pi)*sum(tW*tmu*np.imag(tzp/(tz-z[i])))
				#Compute test sum of monomials, p
				testsum = sum(tW*tzp/(tz-z[i]))

				#if 1:
				if np.abs(p32[0]-testsum) > 1e-13: #Standard 16-GL not good enough!
					#We need to interpolate to 32 point GL
					tmu32 = IPmultR(tmu)
					tz32 = IPmultR(tz)
					tzp32 = IPmultR(tzp)
					plen = tW[0]/W16[0]
					tW32 = W32*plen

					#Compute test sum of monomials, p, with 32GL
					orig32 = tW32/(tz32-z[i])
					o32sum = sum(tzp32*orig32)

					#if 0:
					if np.abs(o32sum-p32[0]) < 1e-13: #32GL suffices!	
						newsum = np.real(1/(2*math.pi)*sum(tW32*tmu32*np.imag(tzp32/(tz32-z[i]))))
						us[i] += (newsum - oldsum)
					else:
						nzpan32 = IPmultR(nzpan)

						signc = -1
						for jl in range(31): #Recursively create p32
							p32[jl+1] = nz*p32[jl] + (1-signc)/(jl+1)
							signc = -signc

						p32c = vandernewton(nzpan32,p32,32)
				
						newsum = 1/(2*math.pi)*sum(np.imag(p32c*tmu32))


						us[i] += (newsum-oldsum)

	return uspecial


    

def vandernewton(T,bin,n):
	""" 
	Compture vandernewton interpolation 
	"""
	b = np.copy(bin)
	for ik in np.arange(1,n):
		ijind = np.arange(ik,n)
		ijind = ijind[::-1]
		for ij in ijind:
			b[ij] -= T[ik-1]*b[ij-1]
	ikind = np.arange(1,n)
	ikind = ikind[::-1]
	for ik in ikind:
		for ij in np.arange(ik,n):
			b[ij] /= T[ij]-T[ij-ik]
			b[ij-1] -= b[ij]
	return b



def vandernewtonT(T,bin,n):
	"""
	Compute vandernewton interpolation with transposed matrix.
	"""
	b = np.copy(bin)
	for ik in range(n-1):
		ijind = np.arange(ik+1,n)
		ijind = ijind[::-1]
		for ij in ijind:
			b[ij] = (b[ij]-b[ij-1])/(T[ij]-T[ij-ik-1])
	ikind = np.arange(0,n)
	ikind = ikind[::-1]
	for ik in ikind:
		for ij in np.arange(ik,n-1):
			b[ij]-= T[ik]*b[ij+1]
	return b

def IPmultR(inv):
	"""
	Interpolate from 16 point GL to 32 point GL
	"""
	outv = np.zeros(32,dtype=np.complex_)
	for ii in range(16):
		t1 = complex(0)
		t2 = complex(0)
		ptr = ii
		for ij in range(8):
			t1 += IP1[ptr]*(inv[ij]+inv[15-ij])
			t2 += IP2[ptr]*(inv[ij]-inv[15-ij])
			ptr += 16
		outv[ii] = t1 + t2
		outv[31-ii] = t1-t2
	return outv




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
	eps = 1e-17
	u2 = u.reshape(np.size(u))
	uc2 = ucorrect.reshape(np.size(ucorrect))
	est = np.abs(u2-uc2)
	est[est<eps] = eps

	us2 = uspecial.reshape(np.size(uspecial))
	espec = np.abs(us2-uc2)
	espec[espec<eps] = eps
	return est, espec


