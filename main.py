import torch as tor
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("",train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))


trainset= tor.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset= tor.utils.data.DataLoader(test,batch_size=10,shuffle=True)

import torch.nn as nn
import torch.nn.functional as fct


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1=nn.Linear(784,64)
		self.fc2=nn.Linear(64,64)
		self.fc3=nn.Linear(64,64)
		self.fc4=nn.Linear(64,10)

	def forward(self, x):
		x=fct.relu(self.fc1(x))
		x=fct.relu(self.fc2(x))
		x=fct.relu(self.fc3(x))
		x=self.fc4(x)
		
		return fct.log_softmax(x,dim=-1)


import torch.optim as optim

net=Net()
print(net)

X=tor.rand((28,28))
X=X.view(28*28)

optimizer=optim.Adam(net.parameters(),lr=0.001)

Epochs=1

for ep in range(Epochs):
	for data in trainset:
		X,y=data
		net.zero_grad()
		output=net(X.view(-1,28*28))
		loss=fct.nll_loss(output,y)
		loss.backward()
		optimizer.step()
	print(loss)

corect=0
total=0

with tor.no_grad():
	for data in testset:
		X,y=data
		output = net(X.view(-1,784))
		for idx,i in enumerate(output):
			if tor.argmax(i) == y[idx]:
				corect+=1
			total+=1
print("Accuracy:",round(corect/total,3))

import matplotlib.pyplot as plt

print(output.size())

plt.imshow(X[0].view(28,28))
plt.show()
print(int(tor.argmax(net(X[0].view(-1,784)))))
