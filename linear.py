import torch
import matplotlib.pyplot as plt  # visual training
import random  # random generate


def create_data(w, b, data_num):
    # create data
    # w(4*1) b(1) data_num(1)
    # x(500*4) y(500*1)
    x = torch.normal(0, 1, (data_num, len(w)))
    y = torch.matmul(x, w) + b  # 'matmul' is an operation for matrix multiplication X(500*4)*W(4*1)+b(1)=Y(500*1)

    noise = torch.normal(0, 0.01, y.shape)
    y += noise

    return x, y


num = 500

true_w = torch.tensor([8.2, 2, 2, 4])
# if here is [8, 2, 2, 4], then w is long type, you need add 'w = w.float()' in create_data()
# make sure the type is the same
true_b = torch.tensor(1.2)

X, Y = create_data(true_w, true_b, num)

# === show result of create_data()
# plt.scatter(X[:, 0], Y, 1)
# plt.show()


def data_provider(data, label, batchsize):
    # provide data as batch
    # data(500*4) label(500*1) batchsize(1)
    # get_data(16*4) get_label(16*1)
    length = len(label)
    indices = list(range(length))

    random.shuffle(indices)  # provide data randomly

    for each in range(0, length, batchsize):
        get_indices = indices[each: each + batchsize]
        get_data = data[get_indices]
        get_label = label[get_indices]

        yield get_data, get_label  # a special 'return' which can save breakpoint


batch_size = 16

# === show result of data_provider()
# for batch_x, batch_y in data_provider(X, Y, batch_size):
#     print(batch_x, batch_y)
#     break


def linear_func(x, w, b):
    # a model to approach true_w true_b
    # x(16*4) w(4*1) b(1)
    # pre_y(16*1)
    pre_y = torch.matmul(x, w) + b
    return pre_y


def mae_loss(pre_y, y):
    # count Mean Absolute Error (MAE) Loss
    # pre_y(16*1) y(16*1)
    # 'result'(1)
    return torch.sum(abs(pre_y - y))/len(y)


def sgd(paras, lr):
    # Stochastic Gradient Descent
    # paras([4*1], 1) lr(1)
    with torch.no_grad():
        # this line is very important, every operation in count will influence gradient
        # so the cycle below would not count grad with this line exist
        for para in paras:
            para -= para.grad * lr
            para.grad.zero_()


lr = 0.01
w_0 = torch.normal(0, 0.01, true_w.shape, requires_grad=True)
b_0 = torch.tensor(0.01, requires_grad=True)
# print(w_0, b_0)

epochs = 50
for epoch in range(epochs):
    data_loss = 0
    for batch_x, batch_y in data_provider(X, Y, batch_size):
        pre_y = linear_func(batch_x, w_0, b_0)
        loss = mae_loss(pre_y, batch_y)
        loss.backward()
        sgd([w_0, b_0], lr)
        data_loss += loss

    print("epoch %03d: loss: %.6f"%(epoch, data_loss))

print("true paras is ", true_w, true_b)
print("learning result is ", w_0, b_0)

idx = 0
plt.plot(X[:, idx].detach().numpy(), X[:, idx].detach().numpy() * w_0[idx].detach().numpy() + b_0.detach().numpy())
plt.scatter(X[:, idx], Y, 1)
plt.show()

# this is my result after first try:
# datasize = 500, batch_size = 16, lr = 0.01, epoch = 50, b_0 = 0.01; without w_0, X, Y
# epoch 000: loss: 245.328278
# epoch 001: loss: 241.125504
# epoch 002: loss: 233.149902
# epoch 003: loss: 224.587097
# epoch 004: loss: 220.882233
# epoch 005: loss: 209.626801
# epoch 006: loss: 204.429794
# epoch 007: loss: 196.281219
# epoch 008: loss: 189.071991
# epoch 009: loss: 185.463470
# epoch 010: loss: 178.510849
# epoch 011: loss: 170.526291
# epoch 012: loss: 164.963333
# epoch 013: loss: 158.164124
# epoch 014: loss: 150.568466
# epoch 015: loss: 144.217590
# epoch 016: loss: 139.919861
# epoch 017: loss: 132.314453
# epoch 018: loss: 125.513046
# epoch 019: loss: 117.942970
# epoch 020: loss: 111.971710
# epoch 021: loss: 106.216324
# epoch 022: loss: 98.260674
# epoch 023: loss: 91.552437
# epoch 024: loss: 85.756523
# epoch 025: loss: 77.237366
# epoch 026: loss: 70.434547
# epoch 027: loss: 64.010544
# epoch 028: loss: 58.121159
# epoch 029: loss: 51.146828
# epoch 030: loss: 45.250122
# epoch 031: loss: 38.659821
# epoch 032: loss: 32.379120
# epoch 033: loss: 25.950603
# epoch 034: loss: 19.586081
# epoch 035: loss: 13.833198
# epoch 036: loss: 8.023759
# epoch 037: loss: 2.374616
# epoch 038: loss: 0.317750
# epoch 039: loss: 0.297884
# epoch 040: loss: 0.298429
# epoch 041: loss: 0.304198
# epoch 042: loss: 0.295577
# epoch 043: loss: 0.297747
# epoch 044: loss: 0.302741
# epoch 045: loss: 0.313951
# epoch 046: loss: 0.304876
# epoch 047: loss: 0.310241
# epoch 048: loss: 0.293368
# epoch 049: loss: 0.326283
# true paras is  tensor([8.2000, 2.0000, 2.0000, 4.0000]) tensor(1.2000)
# learning result is  tensor([8.2010, 1.9937, 1.9930, 3.9886], requires_grad=True) tensor(1.2063, requires_grad=True)
