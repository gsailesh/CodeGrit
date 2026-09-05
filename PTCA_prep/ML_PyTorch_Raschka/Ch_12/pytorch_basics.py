"""
For more operations and functions, refer: https://pytorch.org/docs/stable/index.html.
"""

import torch
import numpy as np
from pprint import pprint

"""
np.set_printoptions(precision=3)
a = [1, 2, 3]

b = np.array([4, 5, 6], dtype=np.int32)
tensor_a = torch.tensor(a)
tensor_b = torch.from_numpy(b)

print(tensor_a)
print(tensor_b)
"""

"""
tensor([1, 2, 3])
tensor([4, 5, 6], dtype=torch.int32)
"""
# ---

"""
tensor_ones = torch.ones(2, 3)
pprint(tensor_ones)
pprint(tensor_ones.shape)
"""

"""
tensor([[1., 1., 1.],
        [1., 1., 1.]])
torch.Size([2, 3])
"""
# ---

"""
# Random tensor
random_tensor = torch.rand(2, 3)
# pprint(random_tensor)
"""

"""
tensor([[0.8371, 0.7017, 0.2363],
        [0.1006, 0.1919, 0.0635]])
"""
# ---
"""
tensor_random_new = random_tensor.to(torch.float16)
pprint(tensor_random_new)
"""

"""
tensor([[0.2319, 0.9570, 0.1648],
        [0.6514, 0.6006, 0.6348]], dtype=torch.float16)
"""
# ---
"""
t = torch.rand(3, 5)
t2 = torch.rand(1, 4, 5)
t_transp = torch.transpose(t, 0, 1)
t_transp_2 = torch.transpose(t2, 2, 0)
t2_reshaped = t2.reshape(2, 10)
t2_squeezed = t2.squeeze() # outer dimension collapses by default
t2_squeezed_alt = t2.squeeze(0) # outer dimension collapses, i.e. zero index
t2_squeezed_alt_2 = t2.squeeze((0, 2)) # outer dimension collapses, but how? ; TODO: Check documentation.

pprint(t.shape)
pprint(t_transp.shape)
pprint(t2.shape)
pprint(t_transp_2.shape)
pprint(t2_reshaped.shape)
pprint(t2_squeezed.shape)
pprint(t2_squeezed_alt.shape)
"""

"""
torch.Size([3, 5])
torch.Size([5, 3])
torch.Size([1, 4, 5])
torch.Size([5, 4, 1])
torch.Size([2, 10])
torch.Size([4, 5])
torch.Size([4, 5])
"""
# ---

"""
tensor_mul = torch.multiply(torch.rand(2, 3), torch.rand(2, 3)) # Element-wise multiplication, dimensions must match.
pprint(tensor_mul) # 2x3

tensor_matmul = torch.matmul(torch.rand(2, 3), torch.rand(3, 2)) # Matrix multiplication, dimensions must match.
pprint(tensor_matmul) # 2x2
"""

"""
tensor([[0.1924, 0.0263, 0.2803],
        [0.6293, 0.1571, 0.1187]])
tensor([[0.6405, 0.4676],
        [0.1363, 0.1178]])
"""

# ---
"""
col_mean = torch.mean(torch.rand(4, 2), axis=0) # outer dimension
row_mean = torch.mean(torch.rand(4, 2), axis=1) # first-dimension or inner dimension in this case

pprint(col_mean.shape) # dim = 2
pprint(row_mean.shape) # dim = 4
"""

"""
torch.Size([2])
torch.Size([4])
"""
# ---

"""
torch.manual_seed(42) # Adds reproducibility
a = torch.rand(2, 3)
norm_tensor = torch.linalg.norm(a, ord=2, dim=1) # dim=1 implies outer dimension
norm_tensor_zerodim = torch.linalg.norm(a, ord=2, dim=0) # dim=0 implies inner-most dimension contrary to axis
norm_tensor_minusone = torch.linalg.norm(a, ord=2, dim=-1) # dim=-1 implies inner-most dimension
pprint(a)
pprint(norm_tensor)
pprint(norm_tensor_zerodim)
pprint(norm_tensor_minusone)
"""

"""
tensor([[0.8823, 0.9150, 0.3829],
        [0.9593, 0.3904, 0.6009]])
tensor([1.3275, 1.1974])
tensor([1.3033, 0.9948, 0.7125])
tensor([1.3275, 1.1974])
"""
# ---
"""
t_split = torch.chunk(torch.rand(2, 5), 2) # TODO: Check documentation for chunk arguments. What is '2'?

for item in t_split:
    pprint(item.numpy().shape) # 2 chunks of dim=(1, 5)
"""

"""
(1, 5)
(1, 5)
"""
# ---

"""
t_another_split = torch.split(torch.rand(3, 5), split_size_or_sections=[1, 2]) # Splits should add up to dim-0 or the outer dimension.
for item in t_another_split:
    pprint(item.numpy().shape) # (1, 5) and (2, 5)
"""

"""
(1, 5)
(2, 5)
"""
# ---

o = torch.ones(3)
z = torch.zeros(2)

a = torch.rand(3, 2)
b = torch.rand(5, 2)

pprint(torch.cat([o, z], axis=0).shape) # concatenates and forms a longer vector, works only for `axis=0` or `axis=-1`
pprint(torch.cat([o, z], axis=-1).shape)

# For arrays
pprint(torch.cat([a, b], axis=0).shape) # tensor sizes must match along the dimension or axis chosen, then it stacks them
pprint(torch.cat([a, b], axis=-2).shape)

# Stacking
pprint(torch.stack([z, z], axis=0).shape) # tensor sizes must be equal
pprint(torch.stack([a, a], axis=0).shape)
pprint(torch.stack([b, b], axis=0).shape)


"""
torch.Size([5])
torch.Size([5])
torch.Size([8, 2])
torch.Size([8, 2])
torch.Size([2, 2])
torch.Size([2, 3, 2])
torch.Size([2, 5, 2])
"""
