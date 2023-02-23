# 作者：ruby
# 开发时间：2022/12/26 19:17
import numpy as np
from PIL import Image
import torch,random
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from data.data_processing import rmlvector,PCA



def project_perturbation(data_point,p,perturbation):
    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.reshape(-1)))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation



def generate(trainset, y_train, trainloader, net, pca=False, max_iter_uni=np.inf, xi=10,p=2, num_classes=2, overshoot=0.2, max_iter_df=20):
    '''
    :param trainset: Pytorch Dataloader with train data
    :param testset: Pytorch Dataloader with test data
    :param net: Network to be fooled by the adversarial examples
    :param delta: 1-delta represents the fooling_rate, and the objective
    :param max_iter_uni: Maximum number of iterations of the main algorithm
    :param p: Only p==2 or p==infinity are supported
    :param num_class: Number of classes on the dataset
    :param overshoot: Parameter to the Deep_fool algorithm
    :param max_iter_df: Maximum iterations of the deep fool algorithm
    :return: perturbation found (not always the same on every run of the algorithm)
    '''

    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Importing images and creating an array with them
    data_trn = trainset
    #     # Setting the number of images to 300  (A much lower number than the total number of instances on the training set)
    #     # To verify the generalization power of the approach
    num_img_trn = 500
    index_order = np.arange(num_img_trn)
    random.seed(151)
    np.random.shuffle(index_order)
    v = np.zeros([2, 128])
    if pca ==False:
        for index in index_order:
            # Generating the original image from data
            cur_data = data_trn[index]
            # print('cur_data:', cur_data.shape)
            cur_data1 = cur_data[np.newaxis, :].to(device)
            # print('cur_data1:',cur_data1.shape)
            # Feeding the original image to the network and storing the label returned
            r2 = (net(cur_data1).max(1)[1])
            torch.cuda.empty_cache()
            # print('r2:', r2)
            # Generating a perturbed image from the current perturbation v and the original image
            v = torch.tensor(v)
            # print('cur_data.shape:',cur_data.shape)
            # print('v.type(torch.float32).shape:',v.type(torch.float32).shape)
            per_data = cur_data + v.type(torch.float32)
            per_data1 = per_data[np.newaxis, :].to(device)
            # print('per_data1:', per_data1.shape)
            # Feeding the perturbed image to the network and storing the label returned
            r1 = (net(per_data1).max(1)[1])
            torch.cuda.empty_cache()
            # print('r1:', r1)
            # If the label of both images is the same, the perturbation v needs to be updated
            if r1 == r2:
                print(">> k =", np.where(index == index_order)[0][0], ', pass #', iter, end='      ')

                # Finding a new minimal perturbation with deepfool to fool the network on this image
                dr, iter_k, label, k_i, pert_data = deepfool(per_data1, net, num_classes=num_classes,
                                                             overshoot=overshoot, max_iter=max_iter_df)

                # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
                if iter_k < max_iter_df - 1:
                    v = v.numpy()
                    v[:, :] += dr[0, 0, :, :]
                    # print('v1:',v)
                    v = project_perturbation(0.5, p, v)
        v= v/np.linalg.norm(v.reshape(-1))

    else:
        tr_data = trainset[index_order]
        tr_y = y_train[index_order]
        dataset = torch.utils.data.TensorDataset(tr_data, tr_y.type(torch.LongTensor))
        loader = DataLoader(dataset, batch_size=num_img_trn, shuffle=False)
        for data, target in loader:
            # Send the data and label to the device
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            output = net(data)
            criterion = nn.CrossEntropyLoss()
            loss =criterion(output, target)
            net.zero_grad()
            loss.backward()
            data_grad = data.grad.data.cpu().numpy().copy()

        data_grad = data_grad/np.linalg.norm(data_grad.reshape(-1))
        trn_vector = rmlvector(data_grad)
        V_r = PCA(trn_vector, 1)
        # v = am* V_r
        # v = v.reshape((-1,2,128))
        v = V_r.reshape((-1, 2, 128))
    return v

def evaluate_performance(accuracy ,testloader, net,v):
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fooling_rates = [0]
    accuracies = []
    accuracies.append(accuracy)
    with torch.no_grad():

        # Compute fooling_rate
        labels_original_data = torch.tensor(np.zeros(0, dtype=np.int64)).to(device)
        labels_pertubed_data = torch.tensor(np.zeros(0, dtype=np.int64)).to(device)
        y_true = torch.tensor(np.zeros(0, dtype=np.int64)).to(device)
        y_pred = torch.tensor(np.zeros(0, dtype=np.int64)).to(device)
        i = 0
        # Finding labels for original images
        for inputs, _ in testloader:
            i += inputs.shape[0]
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted1 = outputs.max(1)
            labels_original_data = torch.cat((labels_original_data, predicted1.to(device)))
        torch.cuda.empty_cache()
        correct = 0

        # Finding labels for perturbed images
        for inputs, labels in testloader:
            # print("inputs:",inputs.dtype)
            inputs += v
            inputs = inputs.type(torch.float32).to(device)
            # print("one inputs:",inputs.dtype)
            outputs = net(inputs)
            _, predicted2 = outputs.max(1)
            labels = labels.to(device)
            labels_pertubed_data = torch.cat((labels_pertubed_data, predicted2))
            y_true = torch.cat((y_true,labels))
            y_pred = torch.cat((y_pred,predicted2))
            # correct += (predicted2 == labels).sum().item()
        correct += (y_pred == y_true).sum().item()
        torch.cuda.empty_cache()
        # Calculating the fooling rate by dividing the number of fooled images by the total number of images
        fooling_rate = float(torch.sum(labels_original_data != labels_pertubed_data)) / float(i)
        # print("FOOLING RATE: ", fooling_rate)
        fooling_rates.append(fooling_rate)
        accuracies.append(correct / i)
        return fooling_rates, accuracies,[y_true,y_pred]

def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()
def deepfool(data,net, num_classes, overshoot=0.02, max_iter=50):

    """
       :param data:
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        data= data.cuda()
        net = net.cuda()


    f_data = net(data).data.cpu().numpy().flatten()
    I = f_data.argsort()[::-1]  # 从小到大排序, 再从后往前复制一遍，So相当于从大到小排序
    I = I[0:num_classes]  # 挑最大的num_classes个(从0开始，到num_classes结束)
    label = I[0]  # 最大的判断的分类
    # label = int(y)
    # index = np.where(I==label)[0][0]
    # other_classes = np.delete(I,index,None)

    input_shape = data.detach().cpu().numpy().shape  # 原始照片
    pert_data = copy.deepcopy(data)  # 干扰照片
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_data, requires_grad=True)
    # net.zero_grad()
    fs = net(x)
    k_i = label

    while k_i == label and loop_i < max_iter:  # 直到分错类别或者达到循环上限次数
        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)  # x产生了grad
        # fs[0, I[index]].backward(retain_graph=True)  # x产生了grad
        grad_orig = x.grad.data.cpu().numpy().copy()  # original grad

        for k in range(1, num_classes):
        # for k in other_classes:
            zero_gradients(x)  # 将梯度置为0
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()  # current grad(分类为k(不是目前所划分的那一类)的grad

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            if np.linalg.norm(w_k.flatten())==0:
                pert_k = abs(f_k) /1e-4
            else:
                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())


            # print(' np.linalg.norm(w_k.flatten()):', np.linalg.norm(w_k.flatten()))
            # determine which w_k to use
            if pert_k < pert:  # 要找到最小的pert
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        if np.linalg.norm(w.flatten()) == 0:
            r_i = pert * w / 1e-4 # 这一次迭代的r
        else:
            # r_i = (pert + 1e-4) * w / (np.linalg.norm(w))
            r_i = pert * w / (np.linalg.norm(w))

        r_tot = np.float32(r_tot + r_i)  # r_total
        if is_cuda:
            pert_data = data + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            pert_data = data + (1 + overshoot) * torch.from_numpy(r_tot)

        x = Variable(pert_data, requires_grad=True)  # 将添加干扰后的图片输入net里面
        fs = net(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())  # 选择最大的一个作为新的分类
        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_data