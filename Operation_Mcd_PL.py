

import numpy as np
import torch
import time
# import ctypes
# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import random
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD, RMSprop
from typing import Optional
import scipy.io as scio
from torch.optim.optimizer import Optimizer
from sklearn import preprocessing
from Adversarial import DomainAdversarialLoss
from model_Mcd_PL import Domain_adaption_model, discriminator, Pairwise_Learning
from DataProcess import get_dataset_cross_session1, get_dataset_cross_allsession, \
    get_true_label_session1, get_true_label_allsession
import warnings
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print("use or no：", torch.cuda.is_available())  #
print("number of GPU：", torch.cuda.device_count())  #
print("Version of CUDA：", torch.version.cuda)  #
print("index of GPU：", torch.cuda.current_device())  #
print("Name of GPU：", torch.cuda.get_device_name(0))  #


# warnings.filterwarnings("ignore")
def setup_seed(seed):  ## setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def discrepancy(out1, out2, threshold=0, if_fea=False):
    if if_fea:
        # out1 = nn.functional.softmax(out1, dim=1)
        # out2 = nn.functional.softmax(out2, dim=1)
        return torch.mean(torch.abs(out1 - out2))


class StepwiseLR_GRL:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


def weigth_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        #        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()


def checkpoint(model, ada_classifity,rms_classifity, checkpoint_PATH, flag, epoch, pred_target,index):  ## saving or loading the checkpoint model
    if flag == 'load':
        pass
        #noNeed
    elif flag == 'save':
        torch.save({'P': model.P, 'stored_mat': model.stored_mat, 'cluster_label': model.cluster_label,
                    'threshold': ada_classifity.threshold,
                    'upper_threshold': ada_classifity.upper_threshold, 'lower_threshold': ada_classifity.lower_threshold,
                    'state_dict': model.state_dict(),#这个是模型的参数
                    'ada_Pairwise_state_dict': ada_classifity.state_dict(),
                    'rms_Pairwise_state_dict': rms_classifity.state_dict(),
                    'target_pred': pred_target}, checkpoint_PATH + '/' +'model'+str(index)+'_'+str(epoch) + '.pth.tar')
        torch.save(model,checkpoint_PATH + '/' +'model'+'_'+str(epoch) + '.pth')
        torch.save(ada_classifity, checkpoint_PATH + '/' + 'ada_Classifity'  + '_' + str(epoch) + '.pth')
        torch.save(rms_classifity, checkpoint_PATH + '/' + 'rms_Classifity'  + '_' + str(epoch) + '.pth')


def plot_heatmap(data_set, best_file, savepath, datapath, tar_set, index):

    label_true = get_true_label_session1(datapath, tar_set) if data_set == 'session1' else get_true_label_allsession(
        datapath, tar_set)

    model_CKPT = torch.load(best_file)
    pred = model_CKPT['target_pred']

    sns.set()
    f, ax = plt.subplots()

    C2 = confusion_matrix(label_true, pred, labels=[0, 1, 2])
    C2 = normalize(C2, axis=1, norm='l1')
    sns.heatmap(C2, annot=True, ax=ax, square=True, cmap='YlGnBu')  # 画热力图

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predict')
    ax.set_ylabel('True Label')
    if index == 0:
        plt.savefig(savepath + '\\last_heatmap_ada_classifity.png')
    else:
        plt.savefig(savepath + '\\best_heatmap_rms_calssisity.png')
    plt.close()


def get_generated_targets(model, ada_classifity, rms_classifity, x_s, x_t,
                          labels_s):  ## Get generated labels by threshold
    with torch.no_grad():
        model.eval()
        _, _, target_feature, _ = model(x_s, x_t, labels_s)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dist_matrix = ada_classifity(target_feature, model.P)
        estimated_sim_truth = model.get_cos_similarity_distance(labels_s)
        sim_matrix_target = ada_classifity.get_cos_similarity_by_threshold(dist_matrix)  # 获取为标签

        dist_matrix1 = rms_classifity(target_feature, model.P)
        sim_matrix_target1 = rms_classifity.get_cos_similarity_by_threshold(dist_matrix1)
        return estimated_sim_truth, sim_matrix_target, sim_matrix_target1



def train_model(loader_train, loader_test,model, ada_classifity,rms_classifity,
                optimizer, optimizer_ada_class, optimizer_rms_class,
                hidden_4, epoch, parameter, batch_num, dann_loss):  ## model training
    cls_loss_sum = 0.0
    transfer_loss_sum = 0.0
    boost_factor = 0.0
    loss_3 = 0.0
    loss = 0.0
    loss_2 = 0.0

    if parameter['boost_type'] == 'linear':
        boost_factor = parameter['cluster_weight'] * (epoch / model.max_iter)
    elif parameter['boost_type'] == 'exp':
        boost_factor = parameter['cluster_weight'] * (2.0 / (1.0 + np.exp(-1 * epoch / model.max_iter)) - 1)
    elif parameter['boost_type'] == 'constant':
        boost_factor = parameter['cluster_weight']
    #Load Data
    train_source_iter, train_target_iter = enumerate(loader_train), enumerate(loader_test)

    # todo:step 1
    for i in range(batch_num):
        # loading data: x_s:source feature,x_t:target feature,labels_s:source label
        model.train()
        ada_classifity.train()
        rms_classifity.train()
        dann_loss.train()
        _, (x_s, labels_s) = next(train_source_iter)
        x_s, labels_s = Variable(x_s.cuda()), Variable(labels_s.cuda())
        _, (x_t, _) = next(train_target_iter)
        x_t = Variable(x_t.cuda())

        # Source domain Truth labels and pseudo labels generated by two classifiers
        estimated_sim_truth, estimated_sim_truth_target, estimated_sim_truth_target1 = get_generated_targets(model,
                                                                                                             ada_classifity,
                                                                                                             rms_classifity,
                                                                                                             x_s, x_t,
                                                                                                             labels_s)
        # sim_matrix: source domain sim matrix
        _, feature_source_f, feature_target_f, sim_matrix = model(x_s, x_t, labels_s)
        eta = 0.00001
        # pairwise loss matrix on source domain
        bce_loss = -(torch.log(sim_matrix + eta) * estimated_sim_truth) - (1 - estimated_sim_truth) * torch.log(
            1 - sim_matrix + eta)
        cls_loss = torch.mean(bce_loss)
        # pairwise loss matrix on target domain
        sim_matrix_target = ada_classifity(feature_target_f, model.P)
        sim_matrix_target_ada = ada_classifity.get_cos_similarity_by_threshold(sim_matrix_target)

        sim_matrix_target1 = rms_classifity(feature_target_f, model.P)
        sim_matrix_target_rms = rms_classifity.get_cos_similarity_by_threshold(sim_matrix_target1)

        # Pairwise loss
        bce_loss_target = -(torch.log(sim_matrix_target_ada + eta) * estimated_sim_truth_target) - \
                          (1 - estimated_sim_truth_target) * torch.log(1 - sim_matrix_target_ada + eta)

        bce_loss_target1 = -(torch.log(sim_matrix_target_rms + eta) * estimated_sim_truth_target1) - \
                           (1 - estimated_sim_truth_target1) * torch.log(1 - sim_matrix_target_rms + eta)

        # valid pair selection for the target domain
        indicator, nb_selected = ada_classifity.compute_indicator(sim_matrix_target)
        cluster_loss = torch.sum(indicator * bce_loss_target) / nb_selected

        indicator1, nb_selected1 = rms_classifity.compute_indicator(sim_matrix_target1)
        cluster_loss1 = torch.sum(indicator1 * bce_loss_target1) / nb_selected1

        # regularization
        P_loss = torch.norm(torch.matmul(model.P.T, model.P) - torch.eye(hidden_4).cuda(), 'fro')
        # domain adversarial loss
        transfer_loss = dann_loss(feature_source_f + 0.005 * torch.randn((parameter['batchsize'], hidden_4)).cuda(),
                                  feature_target_f + 0.005 * torch.randn((parameter['batchsize'], hidden_4)).cuda())

        cls_loss_sum += cls_loss.data  #
        transfer_loss_sum += transfer_loss.data

        loss = cls_loss + transfer_loss * 1 + 0.01 * P_loss + \
               boost_factor * cluster_loss * 1 + boost_factor * cluster_loss1 * 1

        optimizer.zero_grad()
        optimizer_ada_class.zero_grad()
        optimizer_rms_class.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_ada_class.step()
        optimizer_rms_class.step()

    # todo:step 2
    for i in range(batch_num):
        for parm in model.fea_extrator_f.parameters():
            parm.requires_grad = False

        estimated_sim_truth, sim_matrix_target, sim_matrix_target1 = \
            get_generated_targets(model, ada_classifity, rms_classifity, x_s, x_t, labels_s)
        eta = 0.00001
        _, feature_source_f, feature_target_f, sim_matrix = model(x_s, x_t, labels_s)
        bce_loss = -(torch.log(sim_matrix + eta) * estimated_sim_truth) - (1 - estimated_sim_truth) * torch.log(
            1 - sim_matrix + eta)
        cls_loss = torch.mean(bce_loss)
        transfer_loss = dann_loss(feature_source_f + 0.005 * torch.randn((parameter['batchsize'], hidden_4)).cuda(),
                                  feature_target_f + 0.005 * torch.randn((parameter['batchsize'], hidden_4)).cuda())

        dis_target_2 = discrepancy(sim_matrix_target, sim_matrix_target1, if_fea=True)
        loss_2 = transfer_loss + cls_loss - dis_target_2

        optimizer.zero_grad()
        optimizer_ada_class.zero_grad()
        optimizer_rms_class.zero_grad()
        loss_2.backward()
        optimizer.step()
        optimizer_ada_class.step()
        optimizer_rms_class.step()

        for parm in model.fea_extrator_f.parameters():
            parm.requires_grad = True

    # todo:step3
    for i in range(batch_num):

        for parm in ada_classifity.stored_mat:
            parm.detach().requires_grad = False
        for parm1 in ada_classifity.U:
            parm1.detach().requires_grad = False
        for parm2 in ada_classifity.V:
            parm2.detach().requires_gard = False

        for parm in rms_classifity.stored_mat:
            parm.detach().requires_grad = False
        for parm1 in rms_classifity.U:
            parm1.detach().requires_grad = False
        for parm2 in rms_classifity.V:
            parm2.detach().requires_gard = False

        for k in range(3):
            _, feature_source_f, feature_target_f, sim_matrix = model(x_s, x_t, labels_s)
            transfer_loss = dann_loss(feature_source_f + 0.005 * torch.randn((parameter['batchsize'], hidden_4)).cuda(),
                                      feature_target_f + 0.005 * torch.randn((parameter['batchsize'], hidden_4)).cuda())
            #P_loss = torch.norm(torch.matmul(model.P.T, model.P) - torch.eye(hidden_4).cuda(), 'fro')

            bce_loss = -(torch.log(sim_matrix + eta) * estimated_sim_truth) - (1 - estimated_sim_truth) * torch.log(
                1 - sim_matrix + eta)
            cls_loss = torch.mean(bce_loss)
            loss_3 = transfer_loss + cls_loss

            optimizer.zero_grad()
            optimizer_ada_class.zero_grad()
            optimizer_rms_class.zero_grad()
            loss_3.backward()
            optimizer.step()
            optimizer_ada_class.step()
            optimizer_rms_class.step()

        for parm in ada_classifity.stored_mat:
            parm.detach().requires_grad = True
        for parm1 in ada_classifity.U:
            parm1.detach().requires_grad = True
        for parm2 in ada_classifity.V:
            parm2.detach().requires_gard = True

        for parm in rms_classifity.stored_mat:
            parm.detach().requires_grad = True
        for parm1 in rms_classifity.U:
            parm1.detach().requires_grad = True
        for parm2 in rms_classifity.V:
            parm2.detach().requires_gard = True

    print(loss)
    print(loss_2)
    print(loss_3)

    ada_classifity.update_threshold(epoch)
    rms_classifity.update_threshold(epoch)

    return cls_loss_sum.cpu().detach().numpy(), transfer_loss_sum.cpu().detach().numpy()


def train_and_test_GAN(parameter):  ## pipeline for PR-PL model
    setup_seed(parameter['seed'])

    hidden_1, hidden_2, hidden_3, hidden_4 = parameter['hidden_1'], parameter['hidden_2'], \
                                             parameter['hidden_3'], parameter['hidden_4']

    num_of_class, low_rank, upper_threshold, lower_threshold = parameter['num_of_class'], parameter['low_rank'],\
                                                               parameter['upper_threshold'], parameter['lower_threshold']

    if parameter['src_set'] == 'V' or parameter['tar_set'] == 'V':
        lenght_s1 = 6869
        lenght_alls = 18304
    else:
        lenght_s1 = 10035
        lenght_alls = 28350
    batch_num = 0
    if parameter['data_set'] == 'session1':
        batch_num = lenght_s1 // parameter['batchsize']
        target_set, source_set = get_dataset_cross_session1(parameter['datapath'],
                                                            parameter['src_set'], parameter['tar_set'])
    elif parameter['data_set'] == 'allsession':
        batch_num = lenght_alls // parameter['batchsize']
        target_set, source_set = get_dataset_cross_allsession(parameter['datapath'],
                                                              parameter['src_set'], parameter['tar_set'])

    torch_dataset_train = Data.TensorDataset(torch.from_numpy(source_set['feature']),
                                             torch.from_numpy(source_set['label']))
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature']),
                                            torch.from_numpy(target_set['label']))
    test_features, test_labels = torch.from_numpy(target_set['feature']), torch.from_numpy(target_set['label'])
    source_features, source_labels = torch.from_numpy(source_set['feature']), torch.from_numpy(source_set['label'])
    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=parameter['batchsize'],
        shuffle=True,
        num_workers=0
    )
    loader_test = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=parameter['batchsize'],
        shuffle=True,
        num_workers=0
    )
    max_iter = parameter['max_iter']
    setup_seed(parameter['seed'])
    # model initialization
    model = Domain_adaption_model(hidden_1, hidden_2, hidden_4, num_of_class,low_rank, max_iter).cuda(0)
    domain_discriminator = discriminator(hidden_2).cuda()

    ada_classifity = Pairwise_Learning(hidden_3, hidden_4, num_of_class,
                                       low_rank, max_iter, upper_threshold, lower_threshold, model.P).cuda(0)

    rms_classifity = Pairwise_Learning(hidden_2, hidden_4, num_of_class,
                                       low_rank, max_iter, upper_threshold, lower_threshold, model.P).cuda(0)

    model.apply(weigth_init)
    ada_classifity.apply(weigth_init)
    rms_classifity.apply(weigth_init)
    domain_discriminator.apply(weigth_init)
    dann_loss = DomainAdversarialLoss(domain_discriminator).cuda()

    optimizer = RMSprop(model.get_parameters() + domain_discriminator.get_parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer_ada_classifity = Adam(ada_classifity.get_parameters(), lr=0.01, weight_decay=1e-5)
    optimizer_rms_classifity = RMSprop(rms_classifity.get_parameters(), lr=0.01, weight_decay=1e-5)


    best_target_acc = 0.
    best_target_acc1 = 0.
    target_acc_list = np.zeros(max_iter)
    target_nmi_list = np.zeros(max_iter)
    source_acc_list = np.zeros(max_iter)
    source_nmi_list = np.zeros(max_iter)
    target_acc_list1 = np.zeros(max_iter)
    target_nmi_list1 = np.zeros(max_iter)
    cls_loss_list = np.zeros(max_iter)
    transfer_loss_list = np.zeros(max_iter)


    ## model training and evaluation
    for epoch in range(max_iter):
        # train for one epoch
        model.train()
        dann_loss.train()
        ada_classifity.train()
        rms_classifity.train()
        cls_loss_sum, transfer_loss_sum = train_model(loader_train, loader_test,
                                                      model, ada_classifity,rms_classifity,
                                                      optimizer, optimizer_ada_classifity, optimizer_rms_classifity,
                                                      hidden_4, epoch, parameter, batch_num, dann_loss
                                                      )

        model.eval()
        with torch.no_grad():
            source_acc, source_nmi = model.cluster_label_update(source_features.cuda(), source_labels.cuda())

        # target domain evaluation
        feature_target_f = model.fea_extrator_f(test_features.cuda())
        ada_classifity.eval()
        with torch.no_grad():
            target_acc, target_nmi = ada_classifity.target_domain_evaluation(feature_target_f.cuda(),
                                                                             test_labels.cuda())
        rms_classifity.eval()
        with torch.no_grad():
            target_acc1, target_nmi1 = rms_classifity.target_domain_evaluation(feature_target_f.cuda(),
                                                                               test_labels.cuda())

        source_acc_list[epoch] = source_acc
        source_nmi_list[epoch] = source_nmi

        target_acc_list[epoch] = target_acc
        target_nmi_list[epoch] = target_nmi

        target_acc_list1[epoch] = target_acc1
        target_nmi_list1[epoch] = target_nmi1

        cls_loss_list[epoch] = cls_loss_sum
        transfer_loss_list[epoch] = transfer_loss_sum

        best_target_acc = max(target_acc, best_target_acc)
        best_target_acc1 = max(target_acc1, best_target_acc1)

        print(parameter['src_set'], ' to ', parameter['tar_set'], epoch)
        print('s_acc=', source_acc, 's_nmi=', source_nmi)
        print('t_acc=', target_acc, 't_nmi=', target_nmi)
        print('t_acc1=', target_acc1, 't_nmi1=', target_nmi1)
        print('best_target_acc', best_target_acc, '     ', 'best_target_acc1', best_target_acc1)
        print(" ")
        #save the best model
        if target_acc >= best_target_acc:
            feature_target_f = model.fea_extrator_f(test_features.cuda())
            pred_target = ada_classifity.predict(feature_target_f)
            checkpoint(model, ada_classifity, rms_classifity, parameter['savepath'], 'save', epoch, pred_target, 0)
        if target_acc1 >= best_target_acc1:
            feature_target_f = model.fea_extrator_f(test_features.cuda())
            pred_target = rms_classifity.predict(feature_target_f)
            checkpoint(model, ada_classifity, rms_classifity, parameter['savepath'], 'save', epoch, pred_target, 1)

    return best_target_acc, best_target_acc1, cls_loss_list, source_acc_list, source_nmi_list, target_acc_list, target_nmi_list, \
           target_acc_list1, target_nmi_list1, transfer_loss_list


def main(source_data,target_data,source_labels,test_labels):
    fea_extrator_trained = torch.load('model.pth')
    ada_Classifity_trained = torch.load('ada_Classifity.pth')
    rms_Classifity_trained = torch.load('rms_Classifity.pth')
    # Target feature extrator
    _, _, feature_target_f, _ = fea_extrator_trained(source_data,target_data,source_labels)
    # Evaluation
    target_acc_ada, target_nmi_ada = ada_Classifity_trained.target_domain_evaluation(feature_target_f,test_labels)
    target_acc_rms, target_nmi_rms = rms_Classifity_trained.target_domain_evaluation(feature_target_f, test_labels)
    return target_acc_ada, target_nmi_ada, target_acc_rms, target_nmi_rms
