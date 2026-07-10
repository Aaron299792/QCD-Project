class atom:
    def __init__(self, A, R, a, name):
        self.A = A
        self.R = R
        self.a = a
        self.rmax = R + 6.25 * a
        self.name = name

    def print(self):
        print("-"*41)
        print(f"Atom: {self.name}")
        print("-"*41)
        print(f"Atomic Number : {self.A}")
        print(f"Nuclear Radius : {self.R} fm")
        print(f"Nuclear diffusivity constant : {self.a} fm")
        print(f"Maximum radius : {self.rmax:.2f} fm")
        print("-"*41)

if __name__=="__main__":
    atm = atom(208, 6.62, 0.546, "Lead (Pb)")
    atm.print()
