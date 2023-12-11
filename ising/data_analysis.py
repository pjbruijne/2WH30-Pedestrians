import numpy as np

def getCorrelationCoefficient(node:tuple[int,...], lattice:np.ndarray[int,int]) -> float:
    assert len(lattice.shape) == len(node)
    expSpin:float = lattice.sum(dtype=float) / lattice.size
    width, height = lattice.shape
    x:int = np.random.random_integers(np.floor(width/2), np.ceil(width/2))
    y:int = np.random.random_integers(np.floor(height/2), np.ceil(height/2))
    #spinProduct:np.ndarray[int,int] = lattice[x,y] * lattice
    #expSpinProduct:float = spinProduct.sum(dtype=float, keepdims=False) / spinProduct.size
    expSpinProduct:float = lattice[node] * lattice[x,y]
    return expSpinProduct - (expSpin**2)
    #pass
    
def getCorrelationCoefficient_2(node:tuple[int,...], simulation:list[np.ndarray[int,int]]) -> float:
    n = len(node)
    assert len(simulation[0].shape) == len(node)
    expSpin:float = 0
    for i in range(len(simulation)):
        expSpin += simulation[i][node]
    expSpin /= len(simulation)
    if n == 2:
        width, height = simulation[0].shape
        x:int = np.random.randint(np.floor(width/2), np.ceil(width/2)+1)
        y:int = np.random.randint(np.floor(height/2), np.ceil(height/2)+1)
        expSpinProduct:float = 0
        for i in range(len(simulation)):
            expSpinProduct += simulation[i][node] * simulation[i][x,y]
        expSpinProduct /= len(simulation)
        return expSpinProduct - (expSpin**2)
    else: pass