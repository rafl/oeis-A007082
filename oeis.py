from sympy import *
from sympy.polys.matrices import DomainMatrix

init_printing(use_unicode=True)

w, d = symbols('w d')

n = 5
m = 3
p = sum(w**i for i in range(m)) / m

subw = 28
modp = p.subs({w: subw})

def distance(row, col):
    global p
    return ((w ** (2 * row)) * invert((w ** (2 * row)) + (w ** (2 * col)), p))

def fst_term_product(row, col):
    global p
    if (col < row):
        return (w ** (row - col) + invert((w ** (row - col)), p))
    return (w ** (col - row) + invert((w ** (col - row)), p))

def f(multiset):
    mat = []
    numrow = 0
    fst_term = 1
    global p
    for mi in range(len(multiset)):
        for i in range(multiset[mi]):
            row = []
            numcol = 0
            diag_sum = 0
            fst_term *= fst_term_product(mi, 0)
            for mj in range(len(multiset)):
                for j in range(multiset[mj]):
                    if (numrow == numcol):
                        row.append(0)
                        diag_sum += distance(mi, 0)
                    else:
                        dist = distance(mi, mj)
                        row.append(-dist)
                        diag_sum += dist
                        if (numcol > numrow):
                            fst_term *= fst_term_product(mi, mj)
                    numcol += 1
            row[numrow] = diag_sum
            mat.append(row)
            numrow += 1
    mat = Matrix(mat)
    fst_term = factor(fst_term)
    snd_term = mat.det()
    snd_term = factor(snd_term)
    return (fst_term, snd_term, (fst_term * snd_term))

def multiset(l):
    global m 
    m = len(l)
    global n 
    n = sum(l)+1
    return l

# fst, snd, f_n = f(multiset([14,0,0]))
# pretty_print((fst % p).subs({w: subw}))
# pretty_print((snd % p).subs({w: subw}))
# pretty_print((f_n % p).subs({w: subw}))
# pretty_print(Eq(fst, (fst % p).subs({w: subw})))
# pretty_print(Eq(snd, (snd % p).subs({w: subw})))
# pretty_print(Eq(f_n, (f_n % p).subs({w: subw})))

ms1 = [5,0,0]
ms2 = [ms1[(i+1)%m] for i in range(m)]
ms3 = [ms1[(i+2)%m] for i in range(m)]

fst1, snd1, f_n1 = f(ms1)
print(ms1)
pretty_print(f_n1 % modp)
pretty_print(expand(f_n1 % modp) % modp)
pretty_print((f_n1 % p).subs({w: subw}))
# fst2, snd2, f_n2 = f(ms2)
# fst3, snd3, f_n3 = f(ms3)
# print(ms2)
# pretty_print(f_n2 % modp)
# pretty_print(expand(f_n2 % modp) % modp)
# print(ms3)
# pretty_print(f_n3 % modp)
# pretty_print(expand(f_n3 % modp) % modp)

