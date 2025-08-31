'''
Data_process

'''

import numpy as np
import os
import scipy.io as scio
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")



def get_dataset_cross_session1(datapath, src_set, tar_set):  # data loading function for cross dataset
    path_iii = datapath + 'feature_for_net_session' + str(1) + '_LDS_de'
    path_iv = datapath + 'feature_for_net_session' + str(1) + '_LDS_de_IV'
    path_v = datapath + 'feature_for_net_session' + str(1) + '_LDS_de_V'
    feature_list_iii = []
    feature_list_iv = []
    feature_list_v = []
    label_list_iii = []
    label_list_iv = []
    label_list_v = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    os.chdir(path_iii)  # SEED session1
    for info in os.listdir(path_iii):
        domain = os.path.abspath(path_iii)
        info_ = os.path.join(domain, info)
        feature = scio.loadmat(info_)['dataset_session1']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
        feature_list_iii.append(min_max_scaler.fit_transform(feature).astype('float32'))
        #  one-hot label
        one_hot_label_mat = np.zeros((len(label), 3))
        for i in range(len(label)):
            if label[i] == 0:
                one_hot_label = [1, 0, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            elif label[i] == 1:
                one_hot_label = [0, 1, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            elif label[i] == 2:
                one_hot_label = [0, 0, 1]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
        label_list_iii.append(one_hot_label_mat.astype('float32'))

    os.chdir(path_iv)  # SEED_IV session1
    for info in os.listdir(path_iv):
        domain = os.path.abspath(path_iv)
        info_ = os.path.join(domain, info)
        feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset']['label'][0, 0]
        label_ex_fear_index = np.where(label != 2)[0]  # 669 [851(all) - 182(fear)]
        label_ex_fear = label[label_ex_fear_index]
        feature_ex_fear = feature[label_ex_fear_index]

        feature_list_iv.append(min_max_scaler.fit_transform(feature_ex_fear).astype('float32'))
        one_hot_label_mat = np.zeros((len(label_ex_fear), 3))  # IV：exclude fear->2
        for i in range(len(label_ex_fear)):
            if label_ex_fear[i] == 1:  # sad
                one_hot_label = [1, 0, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            elif label_ex_fear[i] == 0:  # neutral
                one_hot_label = [0, 1, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            elif label_ex_fear[i] == 3:  # happy
                one_hot_label = [0, 0, 1]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
        label_list_iv.append(one_hot_label_mat.astype('float32'))

    os.chdir(path_v)  # 读取SEED-V session1
    for info in os.listdir(path_v):
        domain = os.path.abspath(path_v)
        info_ = os.path.join(domain, info)
        feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
        label = scio.loadmat(info_)['dataset']['label'][0, 0]
        # fear
        label_ex_fear_index = np.where(label != 1)[0]
        label_ex_fear = label[label_ex_fear_index]
        feature_ex_fear = feature[label_ex_fear_index]
        # disgust
        label_ex_disgust_index = np.where(label_ex_fear != 0)[0]
        label_ex_fear_disgust = label_ex_fear[label_ex_disgust_index]
        feature_ex_fear_disgust = feature_ex_fear[label_ex_disgust_index]

        feature_list_v.append(min_max_scaler.fit_transform(feature_ex_fear_disgust).astype('float32'))
        # one-hot label
        one_hot_label_mat = np.zeros((len(label_ex_fear_disgust), 3))  # IV：exclude fear->2
        for i in range(len(label_ex_fear_disgust)):
            if label_ex_fear_disgust[i] == 2:  # sad
                one_hot_label = [1, 0, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            elif label_ex_fear_disgust[i] == 3:  # neutral
                one_hot_label = [0, 1, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            elif label_ex_fear_disgust[i] == 4:  # happy
                one_hot_label = [0, 0, 1]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
        label_list_v.append(one_hot_label_mat.astype('float32'))

    if src_set == 'III':
        source_feature, source_label = np.vstack(feature_list_iii), np.vstack(label_list_iii)
    elif src_set == 'IV':
        source_feature, source_label = np.vstack(feature_list_iv), np.vstack(label_list_iv)
    else:
        source_feature, source_label = np.vstack(feature_list_v), np.vstack(label_list_v)

    if tar_set == 'III':
        target_feature, target_label = np.vstack(feature_list_iii), np.vstack(label_list_iii)
    elif tar_set == 'IV':
        target_feature, target_label = np.vstack(feature_list_iv), np.vstack(label_list_iv)
    else:
        target_feature, target_label = np.vstack(feature_list_v), np.vstack(label_list_v)

    target_set = {'feature': target_feature, 'label': target_label}
    source_set = {'feature': source_feature, 'label': source_label}
    return target_set, source_set


def get_dataset_cross_allsession(datapath, src_set, tar_set):  # data loading function for cross dataset
    path_s1_iii = datapath + 'feature_for_net_session' + str(1) + '_LDS_de'
    path_s2_iii = datapath + 'feature_for_net_session' + str(2) + '_LDS_de'
    path_s3_iii = datapath + 'feature_for_net_session' + str(3) + '_LDS_de'
    path_s1_iv = datapath + 'feature_for_net_session' + str(1) + '_LDS_de_IV'
    path_s2_iv = datapath + 'feature_for_net_session' + str(2) + '_LDS_de_IV'
    path_s3_iv = datapath + 'feature_for_net_session' + str(3) + '_LDS_de_IV'
    path_s1_v = datapath + 'feature_for_net_session' + str(1) + '_LDS_de_V'
    path_s2_v = datapath + 'feature_for_net_session' + str(2) + '_LDS_de_V'
    path_s3_v = datapath + 'feature_for_net_session' + str(3) + '_LDS_de_V'

    feature_list_iii = []
    feature_list_iv = []
    feature_list_v = []
    label_list_iii = []
    label_list_iv = []
    label_list_v = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    paths = [path_s1_iii, path_s2_iii, path_s3_iii]
    sessions = [1, 2, 3]

    for path_iii, session in zip(paths, sessions):
        os.chdir(path_iii)  # SEED
        for info in os.listdir(path_iii):
            domain = os.path.abspath(path_iii)
            info_ = os.path.join(domain, info)
            feature = scio.loadmat(info_)['dataset_session' + str(session)]['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session' + str(session)]['label'][0, 0]
            feature_list_iii.append(min_max_scaler.fit_transform(feature).astype('float32'))
            # one-hot label
            one_hot_label_mat = np.zeros((len(label), 3))
            for i in range(len(label)):
                if label[i] == 0:
                    one_hot_label = [1, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
                elif label[i] == 1:
                    one_hot_label = [0, 1, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
                elif label[i] == 2:
                    one_hot_label = [0, 0, 1]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
            label_list_iii.append(one_hot_label_mat.astype('float32'))
        # source_feature, source_label = np.vstack(feature_list_iii), np.vstack(label_list_iii)
        # target_feature, target_label = np.vstack(feature_list_iii), np.vstack(label_list_iii)
    ##########################################################################################################

    for path_iv in [path_s1_iv, path_s2_iv, path_s3_iv]:
        os.chdir(path_iv)  # SEED-IV
        for info in os.listdir(path_iv):
            domain = os.path.abspath(path_iv)
            info_ = os.path.join(domain, info)
            feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset']['label'][0, 0]
            #  fear
            label_ex_fear_index = np.where(label != 2)[0]  # 669 [851(all) - 182(fear)]
            label_ex_fear = label[label_ex_fear_index]
            feature_ex_fear = feature[label_ex_fear_index]

            feature_list_iv.append(min_max_scaler.fit_transform(feature_ex_fear).astype('float32'))
            #  one-hot label
            one_hot_label_mat = np.zeros((len(label_ex_fear), 3))  # IV：exclude fear->2
            for i in range(len(label_ex_fear)):
                if label_ex_fear[i] == 0:  # neutral
                    one_hot_label = [0, 1, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
                elif label_ex_fear[i] == 1:  # sad
                    one_hot_label = [1, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
                elif label_ex_fear[i] == 3:  # happy
                    one_hot_label = [0, 0, 1]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
            label_list_iv.append(one_hot_label_mat.astype('float32'))
        # source_feature, source_label = np.vstack(feature_list_iv), np.vstack(label_list_iv)
        # target_feature, target_label = np.vstack(feature_list_iv), np.vstack(label_list_iv)

    for path_v in [path_s1_v, path_s2_v, path_s3_v]:
        os.chdir(path_v)  # SEED-V
        for info in os.listdir(path_v):
            domain = os.path.abspath(path_v)
            info_ = os.path.join(domain, info)
            feature = scio.loadmat(info_)['dataset']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset']['label'][0, 0]
            # fear
            label_ex_fear_index = np.where(label != 1)[0]
            label_ex_fear = label[label_ex_fear_index]
            feature_ex_fear = feature[label_ex_fear_index]
            #
            label_ex_disgust_index = np.where(label_ex_fear != 0)[0]
            label_ex_fear_disgust = label_ex_fear[label_ex_disgust_index]
            feature_ex_fear_disgust = feature_ex_fear[label_ex_disgust_index]

            feature_list_v.append(min_max_scaler.fit_transform(feature_ex_fear_disgust).astype('float32'))
            #  one-hot label
            one_hot_label_mat = np.zeros((len(label_ex_fear_disgust), 3))  # IV：exclude fear->2
            for i in range(len(label_ex_fear_disgust)):
                if label_ex_fear_disgust[i] == 2:  # sad
                    one_hot_label = [1, 0, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
                elif label_ex_fear_disgust[i] == 3:  # neutral
                    one_hot_label = [0, 1, 0]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
                elif label_ex_fear_disgust[i] == 4:  # happy
                    one_hot_label = [0, 0, 1]
                    one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                    one_hot_label_mat[i, :] = one_hot_label
            label_list_v.append(one_hot_label_mat.astype('float32'))
        # source_feature, source_label = np.vstack(feature_list_v), np.vstack(label_list_v)
        # target_feature, target_label = np.vstack(feature_list_v), np.vstack(label_list_v)
    #
    if src_set == 'III':
        source_feature, source_label = np.vstack(feature_list_iii), np.vstack(label_list_iii)
    elif src_set == 'IV':
        source_feature, source_label = np.vstack(feature_list_iv), np.vstack(label_list_iv)
    else:
        source_feature, source_label = np.vstack(feature_list_v), np.vstack(label_list_v)

    if tar_set == 'III':
        target_feature, target_label = np.vstack(feature_list_iii), np.vstack(label_list_iii)
    elif tar_set == 'IV':
        target_feature, target_label = np.vstack(feature_list_iv), np.vstack(label_list_iv)
    else:
        target_feature, target_label = np.vstack(feature_list_v), np.vstack(label_list_v)

    target_set = {'feature': target_feature, 'label': target_label}
    source_set = {'feature': source_feature, 'label': source_label}
    return target_set, source_set


def get_true_label_session1(datapath, tar_set):
    label_list_tar = []
    if tar_set == 'III':
        path_iii = datapath + 'feature_for_net_session' + str(1) + '_LDS_de'
        os.chdir(path_iii)  # 读取SEED session1
        for info in os.listdir(path_iii):
            domain = os.path.abspath(path_iii)
            info_ = os.path.join(domain, info)
            label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
            label_list_tar.append(label)
        label_tar = np.vstack(label_list_tar)
    elif tar_set == 'IV':
        path_iv = datapath + 'feature_for_net_session' + str(1) + '_LDS_de_IV'
        os.chdir(path_iv)  # 读取SEED-IV session1
        for info in os.listdir(path_iv):
            domain = os.path.abspath(path_iv)
            info_ = os.path.join(domain, info)
            label = scio.loadmat(info_)['dataset']['label'][0, 0]
            # fear 项
            label_ex_fear_index = np.where(label != 2)[0]  # 669 [851(all) - 182(fear)]
            label_ex_fear = label[label_ex_fear_index]
            label_happy_index = np.where(label_ex_fear == 3)[0]
            label_ex_fear[label_happy_index] = 2

            # Label adjustment
            label_neutral_index = np.where(label_ex_fear == 0)[0]
            label_sad_index = np.where(label_ex_fear == 1)[0]
            label_ex_fear[label_neutral_index] = 1
            label_ex_fear[label_sad_index] = 0
            label_list_tar.append(label_ex_fear)
        label_tar = np.vstack(label_list_tar)
    else:
        path_v = datapath + 'feature_for_net_session' + str(1) + '_LDS_de_V'
        os.chdir(path_v)  # SEED-IV session1
        for info in os.listdir(path_v):
            domain = os.path.abspath(path_v)
            info_ = os.path.join(domain, info)
            label = scio.loadmat(info_)['dataset']['label'][0, 0]
            #  fear
            label_ex_fear_index = np.where(label != 1)[0]
            label_ex_fear = label[label_ex_fear_index]
            #disgust
            label_ex_disgust_index = np.where(label_ex_fear != 0)[0]
            label_ex_fear_disgust = label_ex_fear[label_ex_disgust_index]
            #
            label_sad_index = np.where(label_ex_fear_disgust == 2)[0]
            label_neutral_index = np.where(label_ex_fear_disgust == 3)[0]
            label_happy_index = np.where(label_ex_fear_disgust == 4)[0]
            label_ex_fear_disgust[label_sad_index] = 0
            label_ex_fear_disgust[label_neutral_index] = 1
            label_ex_fear_disgust[label_happy_index] = 2
            label_list_tar.append(label_ex_fear_disgust)
        label_tar = np.vstack(label_list_tar)
    return label_tar


def get_true_label_allsession(datapath, tar_set):
    label_list_tar = []
    if tar_set == 'III':
        path_s1_iii = datapath + 'feature_for_net_session' + str(1) + '_LDS_de'
        path_s2_iii = datapath + 'feature_for_net_session' + str(2) + '_LDS_de'
        path_s3_iii = datapath + 'feature_for_net_session' + str(3) + '_LDS_de'
        paths = [path_s1_iii, path_s2_iii, path_s3_iii]
        sessions = [1, 2, 3]
        for path_iii, session in zip(paths, sessions):
            os.chdir(path_iii)  # SEED
            for info in os.listdir(path_iii):
                domain = os.path.abspath(path_iii)
                info_ = os.path.join(domain, info)
                label = scio.loadmat(info_)['dataset_session' + str(session)]['label'][0, 0]
                label_list_tar.append(label)
            label_tar = np.vstack(label_list_tar)
    elif tar_set == 'IV':
        path_s1_iv = datapath + 'feature_for_net_session' + str(1) + '_LDS_de_IV'
        path_s2_iv = datapath + 'feature_for_net_session' + str(2) + '_LDS_de_IV'
        path_s3_iv = datapath + 'feature_for_net_session' + str(3) + '_LDS_de_IV'
        for path_iv in [path_s1_iv, path_s2_iv, path_s3_iv]:
            os.chdir(path_iv)  # 读取SEED-IV
            for info in os.listdir(path_iv):
                domain = os.path.abspath(path_iv)
                info_ = os.path.join(domain, info)
                label = scio.loadmat(info_)['dataset']['label'][0, 0]
                #  fear
                label_ex_fear_index = np.where(label != 2)[0]  # 669 [851(all) - 182(fear)]
                label_ex_fear = label[label_ex_fear_index]
                # happy set 2
                label_happy_index = np.where(label_ex_fear == 3)[0]
                label_ex_fear[label_happy_index] = 2
                # sad and neutral ：label adjustment
                label_neutral_index = np.where(label_ex_fear == 0)[0]
                label_sad_index = np.where(label_ex_fear == 1)[0]
                label_ex_fear[label_neutral_index] = 1
                label_ex_fear[label_sad_index] = 0
                label_list_tar.append(label_ex_fear)
            label_tar = np.vstack(label_list_tar)
    else:
        path_s1_v = datapath + 'feature_for_net_session' + str(1) + '_LDS_de_V'
        path_s2_v = datapath + 'feature_for_net_session' + str(2) + '_LDS_de_V'
        path_s3_v = datapath + 'feature_for_net_session' + str(3) + '_LDS_de_V'
        for path_v in [path_s1_v, path_s2_v, path_s3_v]:
            os.chdir(path_v)  # SEED-V
            for info in os.listdir(path_v):
                domain = os.path.abspath(path_v)
                info_ = os.path.join(domain, info)
                label = scio.loadmat(info_)['dataset']['label'][0, 0]
                #  fear
                label_ex_fear_index = np.where(label != 1)[0]
                label_ex_fear = label[label_ex_fear_index]
                #  disgust
                label_ex_disgust_index = np.where(label_ex_fear != 0)[0]
                label_ex_fear_disgust = label_ex_fear[label_ex_disgust_index]
                label_sad_index = np.where(label_ex_fear_disgust == 2)[0]
                label_neutral_index = np.where(label_ex_fear_disgust == 3)[0]
                label_happy_index = np.where(label_ex_fear_disgust == 4)[0]
                label_ex_fear_disgust[label_sad_index] = 0
                label_ex_fear_disgust[label_neutral_index] = 1
                label_ex_fear_disgust[label_happy_index] = 2

                label_list_tar.append(label_ex_fear_disgust)
            label_tar = np.vstack(label_list_tar)
    return label_tar


