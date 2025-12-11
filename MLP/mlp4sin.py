import torch
import math
import matplotlib.pyplot as plt
import numpy




class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=torch.nn.Linear(1,20)
        self.layer2=torch.nn.Linear(20,20)
        self.layer3=torch.nn.Linear(20,1)

    def forward(self,x):
        x=self.layer1(x)
        x=torch.nn.functional.relu(x)

        x=self.layer2(x)
        x=torch.nn.functional.relu(x)

        x=self.layer3(x)

        return  x



### 
# rnn takes 3d input while mlp only takes 2d input
class RecNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(input_size=1,hidden_size=2,num_layers=1,batch_first=True)
        #至于这个线性层为什么是2维度接收，要看最后网络输出的维度是否匹配label的维度
        self.linear=torch.nn.Linear(2,1)
        
    def forward(self,x):
        # print("x shape: {}".format(x.shape))
        # x [batch_size, seq_len, input_size]
        output,hn=self.rnn(x)
        # print("output shape: {}".format(output.shape))
        # out [seq_len, batch_size, hidden_size]
        x=output.reshape(-1,2)
	
        # print("after change shape: {}".format(x.shape))
        x=self.linear(x)

        # print("after linear shape: {}".format(x.shape))

        return x

def PlotCurve(mlp, rnn, input_x, x):
    # input_x 是输入网络的x。
    # sin_x 是列表，x的取值，一维数据、
    # 虽然他们的内容（不是维度）是一样的。可以print shape看一下。
    mlp_eval = mlp.eval()
    rnn_eval = rnn.eval()
    # mlp_y_np=[]
    # rnn_y_np=[]
    mlp_y = mlp_eval(input_x)
    # for i in range(mlp_y):
    #     mlp_y_np.append(i.detach().numpy())
    rnn_y = rnn_eval(input_x.unsqueeze(0))
    # for i in range(rnn_y):
    #     rnn_y_np.append(i.detach().numpy())
    mlp_y_np=mlp_y.cpu().detach().numpy()
    rnn_y_np=rnn_y.cpu().detach().numpy()

    plt.figure(figsize=(6, 8))

    plt.subplot(211)
    plt.plot([i + 1 for i in range(EPOCH)], mlp_loss, label='MLP')
    plt.plot([i + 1 for i in range(EPOCH)], rnn_loss, label='RNN')
    plt.title('loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(x, torch.sin(x), label="original", linewidth=3)
    plt.plot(x, [y[0] for y in mlp_y_np], label='MLP')
    plt.plot(x, [y[0] for y in rnn_y_np], label='RNN')
    plt.title('evaluation')
    plt.legend()

    plt.tight_layout()
    plt.show()

#常量都取出来，以便改动
EPOCH=1500
RNN_LR=0.01
MLP_LR=0.01
left,right=-2,2
PI=math.pi
NUM=50

if __name__ == '__main__':
    mlp=MLP()
    rnn=RecNN()

    # x,y 是普通sinx 的torch tensor
    x =torch.tensor([(num * PI)  for num in numpy.arange(left, right,(right-left)/NUM)],dtype=torch.float32)
    print(x)
    # x = torch.tensor([num * PI / 4 for num in range(left, right)])
    y = torch.sin(x)
    # input_x和labels是训练网络时候用的输入和标签。
    input_x=x.reshape(-1, 1)
    labels=y.reshape(-1,1)
    input_x=input_x
    labels=labels


    #训练mlp
    mlp_optimizer=torch.optim.Adam(mlp.parameters(), lr=MLP_LR)
    mlp_loss=[]
    for epoch in range(EPOCH):
        preds=mlp(input_x)
        loss=torch.nn.functional.mse_loss(preds,labels)

        mlp_optimizer.zero_grad()
        loss.backward()
        mlp_optimizer.step()
        mlp_loss.append(loss.item())
        # print('1:   ',loss.item())

    #训练rnn
    rnn_optimizer=torch.optim.Adam(rnn.parameters(),lr=RNN_LR)
    rnn_loss=[]
    for epoch in range(EPOCH):
        preds=rnn(input_x.unsqueeze(0))
        # print(x.unsqueeze(0).shape)
        # print(preds.shape)
        # print(labels.shape)
        loss=torch.nn.functional.mse_loss(preds,labels)

        rnn_optimizer.zero_grad()
        loss.backward()
        rnn_optimizer.step()
        rnn_loss.append(loss.item())
        # print('2:   ',loss.item())

    torch.save(mlp,'1.pt')
    PlotCurve(mlp, rnn, input_x, x)

