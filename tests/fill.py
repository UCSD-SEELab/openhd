import openhd as hd

hd.init(D=10, context=globals())
a = hd.hypermatrix(3)
print(a.to_numpy())

hd.fill(a, 1.)
print(a.to_numpy())
