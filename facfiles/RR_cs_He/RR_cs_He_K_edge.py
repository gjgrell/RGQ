from pfac import fac
import math

Zs = ["C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Cr", "Mn", "Fe", "Ni"]
K_edge_He = [3.90848972E+02, 5.50814251E+02, 7.38062856E+02, 1.19452355E+03, 1.76049769E+03, 2.43632608E+03, 3.22242136E+03, 4.11927644E+03, 5.12743621E+03, 7.48036814E+03, 8.13927258E+03, 8.82665243E+03, 1.02873097E+04]

for i in range(len(Zs)):

	fac.MemENTable('/Users/ggrell/software/fac_ions/He-like-adv/'+Zs[i]+'/'+Zs[i]+'02b.en')
	e_list = list(range(math.floor(K_edge_He[i]),20000))
	fac.InterpCross('/Users/ggrell/software/fac_ions/He-like-adv/'+Zs[i]+'/'+Zs[i]+'02b.rr', '/Users/ggrell/software/fac_ions/He-like-adv/RR_cs_He/'+Zs[i]+'02_rr_cs_He_K_edge.txt', 0, -1, e_list, 1)