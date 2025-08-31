import torch

import numpy as np
# import lmmd


class CDD(object):
    def __init__(self, num_layers, kernel_num, kernel_mul, num_classes, threshold, low_rank, hidden_2, intra_only=False):
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only or (self.num_classes==1)
        self.num_layers = num_layers
        self.filtered_classes = []
        self.threshold = threshold
        self.class_num_min = 2
        self.cluster_label = np.zeros(num_classes)
        self.P_tar = torch.zeros(num_classes, 64)
        # self.U = nn.Parameter(torch.randn(low_rank, hidden_2), requires_grad=True)
        # self.V = nn.Parameter(torch.randn(low_rank, hidden_2), requires_grad=True)
        # self.stored_mat = torch.matmul(self.V, self.P_tar.T)
        # self.lmmd = lmmd.LMMD_loss(num_classes)
        self.proto_dist = torch.tensor(0.)
    
    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            dist_c = dist[start:end, start:end]
            dist_list += [dist_c]
        return dist_list

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + 2 * torch.sum(dist['st'])

        bs_S = dist['ss'].size(0)
        bs_T = dist['tt'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N 
        return gamma

    def patch_gamma_estimation(self, nums_S, nums_T, dist):
        assert(len(nums_S) == len(nums_T))
        num_classes = len(nums_S)

        patch = {}
        gammas = {}
        gammas['st'] = torch.zeros_like(dist['st'], requires_grad=False).cuda()
        gammas['ss'] = [] 
        gammas['tt'] = [] 
        for c in range(num_classes):
            gammas['ss'] += [torch.zeros([num_classes], requires_grad=False).cuda()]
            gammas['tt'] += [torch.zeros([num_classes], requires_grad=False).cuda()]

        source_start = source_end = 0
        for ns in range(num_classes):
            source_start = source_end
            source_end = source_start + nums_S[ns]
            patch['ss'] = dist['ss'][ns]

            target_start = target_end = 0
            for nt in range(num_classes):
                target_start = target_end 
                target_end = target_start + nums_T[nt] 
                patch['tt'] = dist['tt'][nt]

                patch['st'] = dist['st'].narrow(0, source_start, nums_S[ns]).narrow(1, target_start, nums_T[nt])

                # y估计
                gamma = self.gamma_estimation(patch)

                gammas['ss'][ns][nt] = gamma
                gammas['tt'][nt][ns] = gamma
                gammas['st'][source_start:source_end, target_start:target_end] = gamma

        return gammas

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):


        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        gamma_tensor = torch.stack(gamma_list, dim=0).cuda()

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps 
        gamma_tensor = gamma_tensor.detach()

        for i in range(len(gamma_tensor.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor
        upper_mask = (dist > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers, key, category=None):
        num_layers = self.num_layers 
        kernel_dist = None
        for i in range(num_layers):

            dist = dist_layers[i][key] if category is None else \
                      dist_layers[i][key][category]

            gamma = gamma_layers[i][key] if category is None else \
                      gamma_layers[i][key][category]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if kernel_dist is None:
                kernel_dist = self.compute_kernel_dist(dist, gamma, cur_kernel_num, cur_kernel_mul)
                continue

            kernel_dist += self.compute_kernel_dist(dist, gamma, cur_kernel_num, cur_kernel_mul)

        return kernel_dist

    def patch_mean(self, nums_row, nums_col, dist):
        assert(len(nums_row) == len(nums_col))
        num_classes = len(nums_row)

        mean_tensor = torch.zeros([num_classes, num_classes]).cuda()
        row_start = row_end = 0
        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0
            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]
                val = torch.mean(dist.narrow(0, row_start, 
                           nums_row[row]).narrow(1, col_start, nums_col[col]))
                mean_tensor[row, col] = val
        return mean_tensor

    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand))**2).sum(2)
        return dist

    def cal_cdd_loss(self, source, target, nums_S, nums_T):
        """
        Args:
            nums_S, nums_T:  example:[300, 300, 300]
        """
        assert(len(nums_S) == len(nums_T)), \
             "The number of classes for source (%d) and target (%d) should be the same." \
             % (len(nums_S), len(nums_T))

        num_classes = len(nums_S)

        # compute the dist 
        dist_layers = []
        gamma_layers = []

        for i in range(self.num_layers):

            cur_source = source
            cur_target = target

            dist = {}
            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)
            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            dist['st'] = self.compute_paired_dist(cur_source, cur_target)

            dist['ss'] = self.split_classwise(dist['ss'], nums_S)
            dist['tt'] = self.split_classwise(dist['tt'], nums_T)


            dist_layers += [dist]

            gamma_layers += [self.patch_gamma_estimation(nums_S, nums_T, dist)]

        # compute the kernel dist
        for i in range(self.num_layers):
            for c in range(num_classes):
                gamma_layers[i]['ss'][c] = gamma_layers[i]['ss'][c].view(num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(num_classes, 1, 1)


        kernel_dist_st = self.kernel_layer_aggregation(dist_layers, gamma_layers, 'st')
        kernel_dist_st = self.patch_mean(nums_S, nums_T, kernel_dist_st)

        kernel_dist_ss = []
        kernel_dist_tt = []
        for c in range(num_classes):

            kernel_dist_ss += [torch.mean(self.kernel_layer_aggregation(dist_layers, gamma_layers, 'ss', c).view(num_classes, -1), dim=1)]
            kernel_dist_tt += [torch.mean(self.kernel_layer_aggregation(dist_layers, gamma_layers, 'tt', c).view(num_classes, -1), dim=1)]

        kernel_dist_ss = torch.stack(kernel_dist_ss, dim=0)
        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)



        mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st
        intra_mmds = torch.diag(mmds, 0)
        intra = torch.sum(intra_mmds) / self.num_classes


        inter = None
        if not self.intra_only:
            a = torch.ones([num_classes, num_classes])
            inter_mask = (a - torch.eye(num_classes)).type(torch.ByteTensor).cuda()
            inter_mmds = torch.masked_select(mmds, inter_mask.bool())
            inter = torch.sum(inter_mmds) / (self.num_classes * (self.num_classes - 1))
        #print(inter)
        #print("")
        cdd = intra if inter is None else intra - inter
        return cdd

    def update_cluster_label(self, cluster_label):
        self.cluster_label = cluster_label

    def update_cdd_threshold(self, threshold):
        self.threshold = threshold

    def fea_label_sort(self, source, target, s_label, t_label):
        """
        Args:
            source,target: Features after sample filtering and classes filtering
            s_label,t_label: Single-label
        """
        sorted_src_fea = []
        sorted_src_labels_num = []
        sorted_tar_fea = []
        sorted_tar_labels_num = []
        if len(self.filtered_classes) == 0:
            pass
        else:
            for cls in self.filtered_classes:
                cls_index = torch.where(s_label == cls)[0]
                sorted_src_fea.append(source[cls_index])
                sorted_src_labels_num.append(cls_index.size(0))

                cls_index_tar = torch.where(t_label == cls)[0]
                sorted_tar_fea.append(target[cls_index_tar])
                sorted_tar_labels_num.append(cls_index_tar.size(0))

            sorted_src_fea = torch.cat(sorted_src_fea)
            sorted_tar_fea = torch.cat(sorted_tar_fea)

        return sorted_src_fea, sorted_tar_fea, sorted_src_labels_num, sorted_tar_labels_num

    def filter_classes(self, t_label):
        """
        Args:
            source,target: Features after sample filtering
            s_label,t_label: Single label
        """
        # if t_label.size()[0] == 0:
        #     return torch.Tensor([0.]).cuda()
        # else:
        self.filtered_classes = []
        for c in range(self.num_classes):
            mask = (t_label == c)
            count = torch.sum(mask).item()
            if count >= self.class_num_min:
                self.filtered_classes.append(c)
            # for i in self.filtered_classes:
            #     filtered_label_index = torch.where(t_label == i)[0]
            #     filtered_label_tar = t_label[filtered_label_index]
            #     filtered_fea_tar = target[filtered_label_index]
            # if filtered_label_tar.size()[0] == 0:
            #     return torch.Tensor([0.]).cuda()
            # else:
            #     self.get_loss(source, filtered_fea_tar, s_label, filtered_label_tar)

    def filter_samples(self, target, t_label):
        """
        Args:
            target:
            t_label: one-hot labels with shape (batch_size * num of classes).
        """
        batch_size_full = int(target.size()[0])
        labels_index = torch.argmax(t_label, dim=1)
        # labels_count = torch.bincount(labels_index)
        labels_single = torch.tensor([t_label[m][labels_index[m]] for m in range(batch_size_full)])

        # min_dist = torch.min(target['dist2center'], dim=1)[0]
        mask = labels_single >= self.threshold

        cluster_0_index, cluster_1_index, cluster_2_index = torch.where(labels_index == 0)[0], \
            torch.where(labels_index == 1)[0], torch.where(labels_index == 2)[0]
        labels_index[cluster_0_index] = self.cluster_label[0]
        labels_index[cluster_1_index] = self.cluster_label[1]
        labels_index[cluster_2_index] = self.cluster_label[2]

        # filtered_feature = torch.tensor([item.cpu().detach().numpy() for item in filtered_feature]).cuda()
        if len([target[m] for m in range(mask.size(0)) if mask[m].item() == 1]) == 0:
            filtered_feature = torch.tensor([])
            filtered_label = torch.tensor([])
            filtered_label_single = torch.tensor([])
        else:
            filtered_feature = torch.stack([target[m] for m in range(mask.size(0)) if mask[m].item() == 1])
            filtered_label_single = torch.stack([labels_index[m] for m in range(mask.size(0)) if mask[m].item() == 1])
            filtered_label = torch.stack([t_label[m] for m in range(mask.size(0)) if mask[m].item() == 1])
        #print(filtered_label_single)
        return filtered_feature.cuda(), filtered_label_single.cuda(), filtered_label_single.size(0), filtered_label


    #loss
    def get_loss(self, source, target, s_label, t_label, source_proto):
        """
        Args:
            source,target:
            s_label,t_label: one-hot labels with shape (batch_size * num of classes).
        """
        # prepare feature and label
        s_label_single = torch.argmax(s_label, dim=1)
        target, t_label_single, selected_num, t_label = self.filter_samples(target, t_label)
        self.filter_classes(t_label_single)
        self.compute_target_proto(target, t_label_single, source_proto)
        source, target, nums_S, nums_T = self.fea_label_sort(source, target, s_label_single, t_label_single)
        # calculate cdd loss
        if len(nums_T) == 0:
            return torch.tensor(0.).cuda(), selected_num, torch.tensor(0.)
        else:
            cdd = self.cal_cdd_loss(source, target, nums_S, nums_T)

            # sim_matrix = torch.clamp(cdd_loss, min=1e-7, max=1 - 1e-7)
            return cdd.cuda(), selected_num, self.proto_dist.cpu()

    def compute_target_proto(self, target, t_label_single, source_proto):
        """
        Args:
            target: (batch_size * hidden_4)
            t_label_single: filtered target label (batch_size)
            source_proto: Domain_adaption_model.P (num of classes, hidden_4)
        """
        if len(self.filtered_classes) == self.num_classes:
            t_label = torch.eye(self.num_classes)[t_label_single].cuda() # to one-hot label
            self.P_tar = torch.matmul(
                torch.inverse(torch.diag(t_label.sum(axis=0)) + torch.eye(self.num_classes).cuda()),
                torch.matmul(t_label.T, target))

            self.proto_dist = self.compute_paired_dist(self.P_tar, source_proto).diag()

            # self.stored_mat = torch.matmul(self.V, self.P_tar.T)
            # target_logit
            # target_predict = torch.matmul(torch.matmul(self.U, target.T).T, self.stored_mat)

