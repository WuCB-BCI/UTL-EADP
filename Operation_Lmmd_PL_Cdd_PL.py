import numpy as np
import time
import torch

import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init

import random
from torch.optim import Adam, SGD, RMSprop
from typing import Optional
from torch.optim.optimizer import Optimizer
from Adversarial import DomainAdversarialLoss
from Basic_Architecture import Domain_adaption_model, discriminator
from DataProcess import get_dataset_cross_session1, get_dataset_cross_allsession
from cdd import CDD
from lmmd import LMMD_loss



def setup_seed(seed):  ## setup the random seed
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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



def get_generated_targets(model, sim_mat_t, labels_s):  ## Get generated labels by threshold
    with torch.no_grad():
        model.eval()
        sim_matrix = model.get_cos_similarity_distance(labels_s)
        sim_matrix_target = model.get_cos_similarity_by_threshold(sim_mat_t)
        return sim_matrix, sim_matrix_target


def checkpoint(model, checkpoint_PATH, flag, epoch, pred_target):  ## saving or loading the checkpoint model
    if flag == 'load':
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        model.P = model_CKPT['P']
        model.stored_mat = model_CKPT['stored_mat']
        model.cluster_label = model_CKPT['cluster_label']
        model.upper_threshold = model_CKPT['upper_threshold']
        model.lower_threshold = model_CKPT['lower_threshold']
        model.threshold = model_CKPT['threshold']
    elif flag == 'save':
        torch.save({'P': model.P, 'stored_mat': model.stored_mat, 'cluster_label': model.cluster_label,
                    'threshold': model.threshold,
                    'upper_threshold': model.upper_threshold, 'lower_threshold': model.lower_threshold,
                    'state_dict': model.state_dict(),
                    'target_pred': pred_target}, checkpoint_PATH + '/' + str(epoch) + '.pth.tar')


def train_model(loader_train, loader_test, model, dann_loss, optimizer, cdd, lmmd,
                epoch, batch_num, writer, record_dir, parameter):
    cls_loss_sum = 0
    transfer_loss_sum = 0
    cluster_loss_sum = 0
    cdd_loss_sum = 0
    lmmd_loss_sum = 0
    total_loss_sum = 0
    P_loss_sum = 0
    eta = 1e-5


    if parameter['boost_type'] == 'linear':
        boost_factor = parameter['cluster_weight'] * (epoch / model.max_iter)
    elif parameter['boost_type'] == 'exp':
        boost_factor = parameter['cluster_weight'] * (2.0 / (1.0 + np.exp(-1 * epoch / model.max_iter)) - 1)
    elif parameter['boost_type'] == 'constant':
        boost_factor = parameter['cluster_weight']

    train_source_iter, train_target_iter = enumerate(loader_train), enumerate(loader_test)
    for i in range(batch_num):

        model.train()
        dann_loss.train()
        _, (x_s, labels_s) = next(train_source_iter)
        x_s, labels_s = Variable(x_s.cuda()), Variable(labels_s.cuda())
        _, (x_t, labels_t) = next(train_target_iter)
        x_t = Variable(x_t.cuda())

        _, feat_s, feat_t, sim_mat_s, sim_mat_t, pred_tar = model(x_s, x_t, labels_s)
        sim_truth_s, sim_truth_t = get_generated_targets(model, sim_mat_t, labels_s)

        feat_src_plot = feat_s.cpu().detach()
        feat_tar_plot = feat_t.cpu().detach()

        bce_loss_s = -(torch.log(sim_mat_s + eta) * sim_truth_s) - (1 - sim_truth_s) * torch.log(1 - sim_mat_s + eta)
        cls_loss_s = torch.mean(bce_loss_s)
        bce_loss_t = -(torch.log(sim_mat_t + eta) * sim_truth_t) - (1 - sim_truth_t) * torch.log(1 - sim_mat_t + eta)
        indicator, nb_selected = model.compute_indicator(sim_mat_t)
        cluster_loss_t = torch.sum(indicator * bce_loss_t) / nb_selected
        P_loss = torch.norm(torch.matmul(model.P.T, model.P) - torch.eye(parameter['hidden_2']).cuda(), 'fro')

        transfer_loss = dann_loss(feat_s + 0.005 * torch.randn((parameter['batchsize'], parameter['hidden_2'])).cuda(),
                                  feat_t + 0.005 * torch.randn((parameter['batchsize'], parameter['hidden_2'])).cuda())

        lmmd_loss = torch.tensor(0.).cuda()
        cdd_loss, selected_cdd, proto_dist = torch.tensor(0.).cuda(), 0, torch.tensor(0.).cuda()

        if parameter['cdd_or_lmmd'] == 'cdd':
            if epoch < parameter['lda_add_epoch']:

                loss = cls_loss_s + transfer_loss + 0.01 * P_loss + boost_factor * cluster_loss_t
            else:

                cdd_loss, selected_cdd, proto_dist = cdd.get_loss(feat_s, feat_t, labels_s, pred_tar, model.P.detach())
                loss = cls_loss_s + transfer_loss + 0.01 * P_loss + boost_factor * cluster_loss_t + \
                       parameter['lda_weight'] * cdd_loss

        elif parameter['cdd_or_lmmd'] == 'lmmd':
            if epoch < parameter['lda_add_epoch']:
                loss = cls_loss_s + transfer_loss + 0.01 * P_loss + boost_factor * cluster_loss_t
            else:

                lmmd_loss = lmmd.get_loss(feat_s, feat_t, labels_s, pred_tar)
                loss = cls_loss_s + transfer_loss + 0.01 * P_loss + boost_factor * cluster_loss_t + \
                       parameter['lda_weight'] * lmmd_loss  #LMMF，直接加的

        else:
            loss = cls_loss_s + transfer_loss + 0.01 * P_loss + boost_factor * cluster_loss_t


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        cls_loss_sum += cls_loss_s.data
        transfer_loss_sum += transfer_loss.data
        cluster_loss_sum += cluster_loss_t.data

        cdd_loss_sum += cdd_loss.data
        lmmd_loss_sum += lmmd_loss.data

        P_loss_sum += P_loss.data
        total_loss_sum += loss.data


    writer.add_scalar('transfer_loss', transfer_loss_sum / batch_num, global_step=epoch)
    writer.add_scalar('selected_for_cdd', selected_cdd, global_step=epoch)
    writer.add_scalar('cdd_loss_sum', cdd_loss_sum / batch_num, global_step=epoch)
    writer.add_scalar('lmmd_loss_sum', cdd_loss_sum / batch_num, global_step=epoch)
    writer.add_scalar('source_loss', cls_loss_sum / batch_num, global_step=epoch)
    writer.add_scalar('target_loss', cluster_loss_sum / batch_num, global_step=epoch)
    writer.add_scalar('total_loss', total_loss_sum / batch_num, global_step=epoch)

    record_dir.update({
        'feat_s': feat_src_plot, 'label_s': labels_s.cpu().detach().numpy(),
        'feat_t': feat_tar_plot, 'label_t': labels_t.cpu().detach().numpy(),
        'cls_loss_sum': cls_loss_sum.cpu().detach().numpy(),
        'transfer_loss_sum': transfer_loss_sum.cpu().detach().numpy(),
        'cluster_loss_sum': cluster_loss_sum.cpu().detach().numpy(),
    })
    record_dir.update({'plotted': False})

    print('transfer_loss:', str(transfer_loss_sum / batch_num), '\np_loss:', str(P_loss_sum / batch_num),
          '\ncdd_loss:', str(cdd_loss_sum / batch_num))

    print('cls_loss_s:', str(cls_loss_sum / batch_num),
          '\ncluster_loss:', str(cluster_loss_sum / batch_num))


    if parameter['update_threshold']:
        threshold = model.update_threshold(epoch)
        if parameter['update_cdd']:
            cdd.update_cdd_threshold(threshold)
    return record_dir


def train_and_test_GAN(parameter):  # pipeline

    if parameter['src_set'] == 'V' or parameter['tar_set'] == 'V':
        lenght_s1 = 6869
        lenght_alls = 18304
    else:
        lenght_s1 = 10035
        lenght_alls = 28350

    if parameter['data_set'] == 'session1':
        batch_num = lenght_s1 // parameter['batchsize']
        target_set, source_set = get_dataset_cross_session1(parameter['datapath'],
                                                            parameter['src_set'], parameter['tar_set'])
    else:
        batch_num = lenght_alls // parameter['batchsize']
        target_set, source_set = get_dataset_cross_allsession(parameter['datapath'],
                                                              parameter['src_set'], parameter['tar_set'])


    torch_dataset_train = Data.TensorDataset(torch.from_numpy(source_set['feature']), torch.from_numpy(source_set['label']))
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature']), torch.from_numpy(target_set['label']))

    test_features, test_labels = torch.from_numpy(target_set['feature']), torch.from_numpy(target_set['label'])
    source_features, source_labels = torch.from_numpy(source_set['feature']), torch.from_numpy(source_set['label'])


    loader_train = Data.DataLoader(dataset=torch_dataset_train, batch_size=parameter['batchsize'], shuffle=True, num_workers=0)
    loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=parameter['batchsize'], shuffle=True, num_workers=0)

    setup_seed(parameter['seed'])

    #model
    model = Domain_adaption_model(parameter['hidden_1'], parameter['hidden_2'],
                                  parameter['hidden_1'], parameter['hidden_2'],
                                  parameter['num_of_class'], parameter['low_rank'], parameter['max_iter'],
                                  parameter['upper_threshold'], parameter['lower_threshold']).cuda()

    model.apply(weigth_init)
    lmmd = LMMD_loss(parameter['num_of_class'])
    cdd = CDD(num_layers=1, kernel_num=(5, 5), kernel_mul=(2, 2), num_classes=parameter['num_of_class'],
              threshold=0.95, low_rank=parameter['low_rank'], hidden_2=parameter['hidden_2'], intra_only=False)


    domain_discriminator = discriminator(parameter['hidden_2']).cuda()
    domain_discriminator.apply(weigth_init)
    dann_loss = DomainAdversarialLoss(domain_discriminator).cuda()

    optimizer = RMSprop(model.get_parameters() + domain_discriminator.get_parameters(), lr=parameter['lr'], weight_decay=1e-5)

    #lr_scheduler = StepwiseLR_GRL(optimizer, init_lr=0.001, gamma=10, decay_rate=0.75, max_iter=parameter['max_iter'])


    record_dir = {'best_target_acc': 0.0}
    target_acc_list = np.zeros(parameter['max_iter'])
    target_nmi_list = np.zeros(parameter['max_iter'])
    source_acc_list = np.zeros(parameter['max_iter'])
    source_nmi_list = np.zeros(parameter['max_iter'])
    cls_loss_list = np.zeros(parameter['max_iter'])
    cluster_loss_list = np.zeros(parameter['max_iter'])
    transfer_loss_list = np.zeros(parameter['max_iter'])

    # model training and evaluation
    for epoch in range(parameter['max_iter']):
        model.train()

        record_dir = train_model(loader_train, loader_test, model, dann_loss, optimizer,
                                 cdd, lmmd,epoch, batch_num, record_dir, parameter)

        source_acc, source_nmi = model.cluster_label_update(source_features.cuda(), source_labels.cuda())
        model.eval()
        target_acc, target_nmi = model.target_domain_evaluation(test_features.cuda(), test_labels.cuda())

        target_acc_list[epoch] = target_acc
        source_acc_list[epoch] = source_acc
        target_nmi_list[epoch] = target_nmi
        source_nmi_list[epoch] = source_nmi
        cls_loss_list[epoch] = record_dir['cls_loss_sum']
        cluster_loss_list[epoch] = record_dir['cluster_loss_sum']
        transfer_loss_list[epoch] = record_dir['transfer_loss_sum']


        if target_acc >= record_dir['best_target_acc']:
            record_dir.update({'best_target_acc': target_acc})

            pred_target = model.predict(test_features.cuda())
            checkpoint(model, parameter['savepath'], 'save', epoch, pred_target)

        print('best_target_acc:', 100.*record_dir['best_target_acc'])


    return record_dir['best_target_acc'], cls_loss_list, transfer_loss_list, cluster_loss_list,\
        source_acc_list, source_nmi_list, target_acc_list, target_nmi_list,


def main(parameter):
    setup_seed(parameter['seed'])
    return train_and_test_GAN(parameter)



