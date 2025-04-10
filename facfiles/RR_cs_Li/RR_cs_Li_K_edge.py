from pfac import fac
import math

Zs = ["C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Cr", "Mn", "Fe", "Ni"]
K_shell_Li = [3.6245E+02, 5.1666E+02, 6.9816E+02, 1.1431E+03, 1.6975E+03, 2.3618E+03, 3.1362E+03, 4.0214E+03, 5.0177E+03, 7.3468E+03, 7.9997E+03, 8.6811E+03, 1.0130E+04]

for i in range(len(Zs)):

	fac.MemENTable('/Users/ggrell/software/fac_ions/Li-like-adv/'+Zs[i]+'/'+Zs[i]+'03b.en')
	e_list = list(range(math.floor(K_shell_Li[i]),20000))
	fac.InterpCross('/Users/ggrell/software/fac_ions/Li-like-adv/'+Zs[i]+'/'+Zs[i]+'03b.rr', '/Users/ggrell/software/fac_ions/Li-like-adv/RR_cs_Li/'+Zs[i]+'03_rr_cs_Li_K_edge.txt', -1, 9, e_list, 1)