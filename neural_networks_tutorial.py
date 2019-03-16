# -*- coding: utf-8 -*-
import torch
import helpers as h
import numpy as np
import random 

def get_batch(df, col='a_odds_ml', batch_size=1):
    batch = df.sample(batch_size)
    batch_labels = batch[col]
    inputs = batch.drop([col], axis=1)
    return inputs, batch_labels


df = h.get_df('data/nba2.csv')
in_shape = len(df.columns) - 1

print(in_shape)
# xtr, ytr = h.split(train_df, 'a_odds_ml')
# xte, yte = h.split(test_df, 'a_odds_ml')

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, in_shape, 3, 1

# Create random Tensors to hold inputs and outputs
x, y = get_batch(df)


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Conv2d(1, D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.middle_linear2 = torch.nn.Linear(H, H)
        self.middle_linear2 = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 6)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

# criterion = torch.nn.MSELoss(reduction='sum')
# model = DynamicNet(D_in, H, D_out)


# # Use the optim package to define an Optimizer that will update the weights of
# # the model for us. Here we will use Adam; the optim package contains many other
# # optimization algoriths. The first argument to the Adam constructor tells the
# # optimizer which Tensors it should update.
# learning_rate = 0.1
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i in range(5000):
#         # get the inputs
#         x, y = get_batch(df)
#         X = torch.from_numpy(np.array(x))
#         Y = torch.from_numpy(np.array(y))
#         # print(X)
#         # zero the parameter gradients


#         # forward + backward + optimize
#         y_pred = model(X)
#         print(str(y_pred) + ' | ' + str(Y))

#         optimizer.zero_grad()
#         loss = criterion(y_pred, Y)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')