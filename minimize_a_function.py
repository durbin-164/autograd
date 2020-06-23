'''
Simple implimentation of minimize a simple function
'''

from autograd.tensor import Tensor, tensor_sum, mul, add

#Simple gradient calculation
x = Tensor(10, requires_grad=True)
y = mul(x, x)
z =  mul(y,y)

z.backward()

print(x.grad)
print(y.grad)

#minimize a function

x = Tensor([10, 5, -10, 5, 2], requires_grad=True)

for i in range(10):
    square_sum = mul(x, x).sum()
    square_sum.backward()

    del_x = mul(Tensor(0.1, requires_grad=True), x.grad)

    x = Tensor(x.data - del_x.data, requires_grad=True)

    print(i, square_sum)
