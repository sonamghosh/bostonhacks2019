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


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    test = pd.read_csv('./dataset/Boston-University_tuition.csv')

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
    p = np.poly1d(forecast)  # forecasted equation

    xp = np.arange(20)


    timestamps = test[['YearDT']]
    timestamps = timestamps.YearDT.tolist()
    timestamps = [time[0:4] for time in timestamps]
    num_pts = len(xp) - len(x_train)

    max_time = timestamps[-1]
    for pt in range(1, num_pts+1):
        val = int(max_time) + pt
        timestamps.append(str(val))
    print(timestamps)

    #plt.plot(xp, p(xp), '*')
    plt.clf()
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
    plt.plot(timestamps, p(xp), 'ro', label='Forecasted', alpha=0.5)
    plt.plot(timestamps[-num_pts-1:], p(xp[-num_pts-1:]), 'r--', alpha=0.5)
    plt.title('Forecasted Cost for Boston University for {num_pts} years from the year {max_time}'.format(num_pts=str(num_pts), max_time=max_time))
    plt.legend(loc='best')
    plt.show()



    """
    xt = range(predicted.shape[0])
    fit = p(xt)

    # Coordinates of fit curve
    c_y = [np.min(fit), np.max(fit)]
    c_x = [np.min(xt), np.max(xt)]

    # predict y value of original date using fit
    p_y = forecast[0] * xt + forecast[1]

    # Calculate y-error (resids)
    y_err = y_train - p_y

    # create series of new test x-values to predict for
    p_x = np.arange(np.min(xt),np.max(xt)+1,1)

    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(xt)         # mean of x
    n = len(xt)              # number of samples in origional fit
    t = 2.31                # appropriate t value (where n=9, two tailed 95%)
    s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

    confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/
                ((np.sum(np.power(xt,2)))-n*(np.power(mean_x,2))))))

    # now predict y based on test x-values
    p_y = forecast[0]*p_x+forecast[0]

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs)
    upper = p_y + abs(confs)

    # set-up the plot
    plt.axes().set_aspect('equal')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Linear regression and confidence limits')

    # plot sample data
    plt.plot(x_train,predicted,'bo',label='Sample observations', alpha=0.5)

    # plot line of best fit
    plt.plot(c_x,c_y,'r-',label='Regression line')

    # plot confidence limits
    plt.plot(p_x,lower,'b--',label='Lower confidence limit (95%)')
    plt.plot(p_x,upper,'b--',label='Upper confidence limit (95%)')

    # set coordinate limits
    plt.xlim(0,11)
    plt.ylim(0,11)

    # configure legend
    plt.legend(loc=0)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10)

    # show the plot
    plt.show()
    """
