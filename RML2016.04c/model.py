# 作者：ruby
# 开发时间：2022/12/26 20:29
import torch.nn as nn
import torch.nn.functional as F
class ConvNet128(nn.Module):
    def __init__(self,classes):
        super(ConvNet128, self).__init__()
        dr =0.6
        self.conv1 = nn.Conv2d(1, 256, (1, 7), padding=(0, 3))
        self.b1 = nn.BatchNorm2d(256)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.dropout1 = nn.Dropout(dr)
        self.zeropad2 = nn.ZeroPad2d((0, 2))
        self.conv2 = nn.Conv2d(256, 128, (1, 7), padding=(0, 3))
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        self.b2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(dr)
        self.conv3 = nn.Conv2d(128, 80, (2, 7), padding=(0, 3))
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))
        self.b3 = nn.BatchNorm2d(80)
        self.dropout3 = nn.Dropout(dr)
        self.fc1 = nn.Linear(80 * 128, 256)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(256, 128)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.dropout4 = nn.Dropout(dr)
        self.fc3 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        self.fc4 = nn.Linear(64, classes)
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='sigmoid')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.b1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.b2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.b3(x)
        x = x.view(-1, 80 * 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        return out



