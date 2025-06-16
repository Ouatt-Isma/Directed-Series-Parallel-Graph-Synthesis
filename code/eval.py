import numpy as np 
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16, })
class TO:
    def __init__(self, b=1/3, d=1/3, u=1/3, a=0.5):
        assert  b>=0 and d>=0 and u>=0, f"{b},{d},{u}"
        assert  round(b+d+u)==1, f"{b},{d},{u}"
        self.b = b
        self.d = d
        self.u = u
        self.a = a 
        

def disc(op1, op2):
    a2 = op2.a
    a = a2
    e1 = op1.b + op1.u*op1.a 
    b2 = op2.b 
    d2 = op2.d 
    b = e1 * b2
    d = e1 * d2
    u = 1 - b - d
    return TO(b,d,u,a)


def fusion(op1, op2):
    b1, u1, a1 = op1.b, op1.u, op1.a 
    b2, u2, a2 = op2.b, op2.u, op2.a 
    if u1 != 0 or u2 != 0:
        denom = u1 + u2 - u1 * u2
        b = (b1 * u2 + b2 * u1) / denom
        u = u1 * u2 / denom
        if u1 != 1 or u2 != 1:
            a = (a1 * (1 - u1) + a2 * (1 - u2)) / (2 - u1 - u2)
        else:
            a = (a1 + a2) / 2
    else:
        b = 0.5 * (b1 + b2)
        u = 0
        a = 0.5 * (a1 + a2)

    d = 1 - u - b
    return TO(b,d,u,a)


def averaging_fusion(op1, op2):
    b1, u1, a1 = op1.b, op1.u, op1.a 
    b2, u2, a2 = op2.b, op2.u, op2.a 

    if u1 != 0 or u2 != 0:
        b = (b1 * u2 + b2 * u1) / (u1 + u2)
        u = 2 * u1 * u2 / (u1 + u2)
        a = (a1 + a2) / 2
    else:
        b = 0.5 * (b1 + b2)
        u = 0
        a = 0.5 * (a1 + a2)

    d = 1 - b - u
    e = b + a * u

    return TO(b,d,u,a)


def compute_full_expression(omega_AB, omega_BC, omega_CE, omega_BE, omega_AD, omega_DE, fusion=fusion):
    inner = fusion(disc(omega_BC, omega_CE), omega_BE)
    left = disc(omega_AB, inner)
    right = disc(omega_AD, omega_DE)
    result = fusion(left, right)
    return result

def compute_reduced_expression(omega_AB, omega_BC, omega_CE, omega_AD, omega_DE, fusion=fusion):
    left = disc(omega_AB, disc(omega_BC, omega_CE))
    right = disc(omega_AD, omega_DE)
    result = fusion(left, right)
    return result

def main1():
    omega_AB, omega_BC, omega_CE, omega_AD, omega_DE = TO(), TO(), TO(), TO(), TO()
    # omega_AB, omega_BC, omega_CE, omega_AD, omega_DE = TO(b=0.5, d=0.2, u=0.3), TO(), TO(b=0.5, d=0.2, u=0.3), TO(), TO(b=0.5, d=0.2, u=0.3)
    B = np.linspace(0, 0.5, 100)
    X = 1 - 2*np.linspace(0, 0.5, 100)
    Y1 = []
    Y2 = []
    for b in B:
        omega_BE = TO(b,b,1-2*b)
        tmp = compute_full_expression(omega_AB, omega_BC, omega_CE, omega_BE, omega_AD, omega_DE)
        Y1.append(tmp.u)
        tmp = compute_reduced_expression(omega_AB, omega_BC, omega_CE, omega_AD, omega_DE)
        Y2.append(tmp.u)
    plt.plot(X,  Y1, label="without relationship B-E")
    plt.plot(X,  Y2, label="with relationship B-E")
    plt.xlabel(r'uncertainty of $\omega^B_E$')
    plt.ylabel(r'uncertainty of the derived $\omega^A_E$')
    yticks = plt.yticks()[0].tolist()
 
    yticks.append(0.6)
    yticks = sorted(yticks)
    plt.yticks(yticks)
    plt.legend()
    plt.savefig("evalCum.pdf", bbox_inches='tight')
    plt.clf() 
   
def main2():
    omega_AB, omega_BC, omega_CE, omega_AD, omega_DE = TO(), TO(), TO(), TO(), TO()
    # omega_AB, omega_BC, omega_CE, omega_AD, omega_DE = TO(b=0.5, d=0.2, u=0.3), TO(), TO(b=0.5, d=0.2, u=0.3), TO(), TO(b=0.5, d=0.2, u=0.3)
    B = np.linspace(0, 0.5, 100)
    X = 1 - 2*np.linspace(0, 0.5, 100)
    Y1 = []
    Y2 = []
    for b in B:
        omega_BE = TO(b,b,1-2*b)
        tmp = compute_full_expression(omega_AB, omega_BC, omega_CE, omega_BE, omega_AD, omega_DE, averaging_fusion)
        Y1.append(tmp.u)
        tmp = compute_reduced_expression(omega_AB, omega_BC, omega_CE, omega_AD, omega_DE, averaging_fusion)
        Y2.append(tmp.u)
    plt.plot(X,  Y1, label="without relationship B-E")
    plt.plot(X,  Y2, label="with relationship B-E")
    plt.xlabel(r'uncertainty of $\omega^B_E$')
    plt.ylabel(r'uncertainty of the derived $\omega^A_E$')
    # yticks = plt.yticks()[0].tolist()
 
    # yticks.append(0.6)
    # yticks = sorted(yticks)
    # plt.yticks(yticks)
    plt.legend()
    plt.savefig("evalAvg.pdf", bbox_inches='tight')
    plt.clf() 
    
def main():
    omega_AB, omega_BC, omega_CE, omega_AD, omega_DE = TO(), TO(b=0.5, d=0.2, u=0.3), TO(), TO(b=0.5, d=0.2, u=0.3), TO(b=0.5, d=0.2, u=0.3)
    B = np.linspace(0.1, 0.9, 100)
    X = B/(1-B)
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    for b in B:
        omega_BE = TO(b,1-b,0)
        tmp = compute_full_expression(omega_AB, omega_BC, omega_CE, omega_BE, omega_AD, omega_DE)
        Y1.append(tmp.u)
        Y3.append(tmp.d)
        tmp = compute_reduced_expression(omega_AB, omega_BC, omega_CE, omega_AD, omega_DE)
        Y2.append(tmp.u)
        Y4.append(tmp.d)
    plt.plot(X,  Y1, label="without relationship B-E")
    plt.plot(X,  Y2, label="with relationship B-E")
    # plt.plot(X,  Y3, label="disbelief without relationship B-E")
    # plt.plot(X,  Y4, label="disbelief with relationship B-E")
    plt.xlabel(r'belief of $\omega^B_E$')
    plt.ylabel(r'Uncertainty of $\omega^A_E$')

    plt.legend()
    plt.savefig("eval2.pdf", bbox_inches='tight')
main2()
main1()