import multiprocessing as mp

def foo(para):
    return para,

pool = mp.Pool(mp.cpu_count()) 
results = pool.map(foo,[i for i in range(5)])
print(results)