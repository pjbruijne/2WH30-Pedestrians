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
    
def getCorrelationCoefficient_3(simulation:list[np.ndarray[int,int]], useWrap:bool|None=False) -> np.ndarray[float]:
    """
    Based on the stackexchange post 
    https://physics.stackexchange.com/questions/169442/how-to-measure-the-spin-spin-correlation-in-a-monte-carlo-simulation-of-the-isin.\n
    Assumes that the lattice is square, not simply rectangular.
    """
    width, height = simulation[0].shape
    assert width == height
    L = width
    result = np.zeros(L,dtype=np.float64)
    for lattice in simulation:
        for r in range(L):
            samples = 0
            sum = 0
            for i in range(L):   
                if r <= i:
                    sum += lattice[i,i]*(lattice[i-r,i] + lattice[i,i-r])
                    samples += 2
                if i < L-r:
                    sum += lattice[i,i]*(lattice[i+r,i] + lattice[i,i+r])
                    samples += 2
            result[r] += sum / samples
    result /= len(simulation)-1
    return result

def getCorrelationCoefficient_4(node:tuple[int,int], simulation:list[np.ndarray[int,int]]) -> np.ndarray[float]:
    width, height = simulation[0].shape
    x,y = node
    maxR = max(abs(width-x),abs(height-y))
    result = np.zeros(maxR,dtype=np.float64)
    meanS0 = 0
    meanS1 = np.zeros(maxR, dtype=np.float64)
    meanS1_samples = 0
    for lattice in simulation:
        meanS0 += lattice[x,y] 
        for r in range(maxR):
            samples = 0
            sum = 0 
            if r <= x:
                sum += lattice[x,y]*lattice[x-r,y]
                meanS1[r] += lattice[x-r,y]
                samples += 1
                meanS1_samples += 1
            if r <= y:
                sum += lattice[x,y]*lattice[x,y-r]
                meanS1[r] += lattice[x,y-r]
                samples += 1
                meanS1_samples += 1
            if x < width-r:
                sum += lattice[x,y]*lattice[x+r,y]
                meanS1[r] += lattice[x+r,y]
                samples += 1
                meanS1_samples += 1
            if y < height - r:
                sum += lattice[x,y]*lattice[x,y+r]
                meanS1[r] += lattice[x,y+r]
                samples += 1
                meanS1_samples += 1
            result[r] += sum / samples
    result /= len(simulation)-1
    meanS0 /= len(simulation)
    meanS1 /= meanS1_samples
    result2 = result - meanS0 * meanS1
    return result,result2