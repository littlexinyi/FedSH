import torch
import numpy as np

def calculate_similarity(image_feature_local, text_feature_local):

      image_feature_local = image_feature_local.view(image_feature_local.size(0), -1)
      image_feature_local = image_feature_local / image_feature_local.norm(dim=1, keepdim=True)

      text_feature_local = text_feature_local.view(text_feature_local.size(0), -1)
      text_feature_local = text_feature_local / text_feature_local.norm(dim=1, keepdim=True)

      similarity = torch.mm(image_feature_local, text_feature_local.t())
      return similarity.cpu()

def calculate_ap(similarity, label_query, label_gallery):
    """
        calculate the similarity, and rank the distance, according to the distance, calculate the ap, cmc
    :param label_query: the id of query [1]
    :param label_gallery:the id of gallery [N]
    :return: ap, cmc
    """

    index = np.argsort(similarity)[::-1]  # the index of the similarity from huge to small
    good_index = np.argwhere(label_gallery == label_query)  # the index of the same label in gallery

    cmc = np.zeros(index.shape)

    mask = np.in1d(index, good_index)  # get the flag the if index[i] is in the good_index

    precision_result = np.argwhere(mask == True)  # get the situation of the good_index in the index

    precision_result = precision_result.reshape(precision_result.shape[0])

    if precision_result.shape[0] != 0:
        cmc[int(precision_result[0]):] = 1  # get the cmc

        d_recall = 1.0 / len(precision_result)
        ap = 0

        for i in range(len(precision_result)):  # ap is to calculate the PR area
            precision = (i + 1) * 1.0 / (precision_result[i] + 1)

            if precision_result[i] != 0:
                old_precision = i * 1.0 / precision_result[i]
            else:
                old_precision = 1.0

            ap += d_recall * (old_precision + precision) / 2

        return ap, cmc
    else:
        return None, None

def evaluate(similarity, label_query, label_gallery):
    similarity = similarity.numpy()
    label_query = label_query.numpy()
    label_gallery = label_gallery.numpy()

    cmc = np.zeros(label_gallery.shape)
    ap = 0
    for i in range(len(label_query)):
        ap_i, cmc_i = calculate_ap(similarity[i, :], label_query[i], label_gallery)
        cmc += cmc_i
        ap += ap_i
    """
    cmc_i is the vector [0,0,...1,1,..1], the first 1 is the first right prediction n,
    rank-n and the rank-k after it all add one right prediction, therefore all of them's index mark 1
    Through the  add all the vector and then divive the n_query, we can get the rank-k accuracy cmc
    cmc[k-1] is the rank-k accuracy   
    """
    cmc = cmc / len(label_query)
    map = ap / len(label_query)  # map = sum(ap) / n_query

    # print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (cmc[0], cmc[4], cmc[9], map))

    return cmc, map