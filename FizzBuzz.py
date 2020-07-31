import numpy as np
import torch

NumDigit = 10

def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0

def fizz_buzz_decode(i, prediction):
# 形成一个列表，str(i)， 把当前数字转换成字符串，放在0的位置，fizz_buzz_encode(i)默认返回值是0，输出当前数字的字符串形式。
    return [str(i), 'fizz', 'buzz', 'fizzbuzz'][prediction]


def binary_encode(i, NumDigit):
    return np.array([i >> d & 1 for d in range(NumDigit) ][::-1])

trX = torch.Tensor([binary_encode(i, NumDigit) for i in range(101,2**NumDigit)])
# 注意是LongTensor， 因为Tensor默认数据类型为float，而fizz_buzz_encode(i)返回数据类型为整型数据，所以用LongTensor
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NumDigit)])

binary_encode(101, NumDigit)

num_hiddens = 100
model = torch.nn.Sequential(
    torch.nn.Linear(NumDigit, num_hiddens),
    torch.nn.ReLU(),
    torch.nn.Linear(num_hiddens, 4),
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 下面训练模型
batch_size = 128
for epoch in range(10000):
    for start in range(0, len(trX), batch_size):
        end = start + batch_size
        batchX = trX[start:end]
        batchY = trY[start:end]

        y_pred = model(batchX)  # forward
        loss = loss_fn(y_pred, batchY)  # 分类的损失函数
        print("Epoch:", epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()  # backpass
        optimizer.step()  # gradient descent

# 测试模型如何,101到1024用作了训练，1到100用作测试
testX = torch.Tensor([binary_encode(i, NumDigit) for i in range(1,101)])
# 测试不需要梯度，可以防止内存占用过高而爆掉
with torch.no_grad():
    testY = model(testX)
    print(testY)
    print('%'*80)
    print(testY.max(1))  # 注意max后面的括号
    print('%'*80)
    print(testY.max(1)[1])  # max(1)后面的括号

predictions = zip(range(1,101),testY.max(1)[1].tolist())  #
# for x,y in predictions:
#     print(x,y)

# print([x,y for x,y in predictions])
print([fizz_buzz_decode(i, x) for i, x in predictions])


