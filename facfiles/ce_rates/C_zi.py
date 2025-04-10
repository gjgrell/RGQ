from pfac import fac
import math

#Zs = ["C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Cr", "Mn", "Fe", "Ni"]
#K_edge_He = [3.90848972E+02, 5.50814251E+02, 7.38062856E+02, 1.19452355E+03, 1.76049769E+03, 2.43632608E+03, 3.22242136E+03, 4.11927644E+03, 5.12743621E+03, 7.48036814E+03, 8.13927258E+03, 8.82665243E+03, 1.02873097E+04]

Zs = ["O"]
K_edge_He = [7.38062856E+02]
#6.53284902E+01
for i in range(len(Zs)):

	fac.MemENTable('/Users/ggrell/software/fac_ions/He-like-adv/'+Zs[i]+'/'+Zs[i]+'02b.en')
# 	e_list = list(range(math.floor(K_edge_He[i]),1000))
	e_list = list(range(1,10000))
	fac.InterpCross('/Users/ggrell/software/fac_ions/He-like-adv/'+Zs[i]+'/'+Zs[i]+'02b.ce', '/Users/ggrell/software/fac_ions/He-like-adv/ce_rates/'+Zs[i]+'02_ce_cs.txt', 1, 3, e_list, 1)
	

# #Array to store CE cross sections
# CE_cs = []

# 
# for i in range(len(Zs)):
# 
#     egrid = []
#     ce_sig = []
# 
#     # Opens input file
#     data = np.loadtxt('/Users/ggrell/software/fac_ions/He-like-adv/ce_rates/'+Zs[i]+'02_ce_cs.txt',skiprows=1,usecols=(0,3,4))
# 
#     for i in range(len(data)):
#         egrid.append(data[i][0])
#         ce_sig.append(data[i][1])
# # 
#     ce_cs = simps(ce_sig, egrid)
#     CE_cs.append(ce_cs)
# 
# print(CE_cs)