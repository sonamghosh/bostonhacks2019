import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Linear Regressor
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# LSTM
class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1)
        return outputs


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    test = pd.read_csv('./dataset/Northeastern-University_tuition.csv')

    df = test[['TUITIONFEE_OUT']]

    df.columns = ['tuition']

    data = df.tuition.tolist()

    y_train = np.array(data, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    x_vals = list(range(len(data)))
    x_train = np.array(x_vals, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)


    #torch.save(data, open('traindata.pt', 'wb'))

    input_dim = 1
    output_dim = 1

    model = LinearRegressionModel(input_dim, output_dim)

    criterion = nn.MSELoss()
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epochs = 1000

    for epoch in range(epochs):
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

        # Clear gradient buffers
        optimizer.zero_grad()

        # get output
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        print(loss)
        # gradient w.r.t params
        loss.backward()
        # update
        optimizer.step()


        print('epoch {}, loss {}'.format(epoch, loss.item()))

    with torch.no_grad():
        predicted = model(Variable(torch.from_numpy(x_train)))

    print(predicted.numpy().reshape(1, predicted.shape[0])[0])
    predicted_arr = predicted.numpy().reshape(1, predicted.shape[0])[0]
    forecast = np.polyfit(list(range(len(predicted_arr))), predicted_arr, 1)

    print(forecast)
    p = np.poly1d(forecast)

    xp = np.arange(20)
    plt.plot(xp, p(xp), '*')
    #plt.clf()
    #plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    #plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
    #plt.legend(loc='best')
    plt.show()

    """
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    # train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()
    """
