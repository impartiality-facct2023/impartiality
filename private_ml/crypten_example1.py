import torch
import crypten

crypten.init()

x = torch.tensor([1.0, 2.0, 3.0])
print('x: ', x)
x_enc = crypten.cryptensor(x)  # encrypt

x_dec = x_enc.get_plain_text()  # decrypt

y = torch.tensor([2.0, 3.0, 4.0])
print('y: ', y)
y_enc = crypten.cryptensor(y)

sum_xy = x_enc + y_enc  # add encrypted tensors
sum_xy_dec = sum_xy.get_plain_text()  # decrypt sum
print('x + y: ', sum_xy_dec)

z = torch.tensor([5.0, 7.0, 2.0])

sum_xz = x_enc + z
sum_xz_dec = sum_xz.get_plain_text()
print('x + z: ', sum_xz_dec)
