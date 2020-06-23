'''
Simple implimentation of minimize a simple function
'''

from autograd.tensor import Tensor

#Simple gradient calculation
x = Tensor(10, requires_grad=True)
y = x * x
z =  y * y

z.backward()

print(x.grad)
print(y.grad)

#minimize a function

x = Tensor([10, 5, -10, 5, 2], requires_grad=True)

for i in range(10):
    x.zero_grad()
    square_sum = (x * x).sum()
    square_sum.backward()

    del_x = 0.1 * x.grad

    x -= del_x

    print(i, square_sum)
