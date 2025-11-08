import torch




def cos_distance(feature1, feature2):
    feature1 = feature1.reshape(-1, feature1.size(-1))
    feature2 = feature2.reshape(-1, feature2.size(-1))
    return torch.sum(feature1 * feature2, -1) / (torch.norm(feature1) * torch.norm(feature2))

def batch_cos_distance(feature1, feature2):
    batch_size = feature1.size(0)
    distances = []
    for i in range(batch_size):
        distances.append(cos_distance(feature1[i], feature2[i]))
    return torch.cat(distances)


def E_distance(feature1,feature2):
    feature1 = feature1.reshape(-1, feature1.size(-1))
    feature2 = feature2.reshape(-1, feature2.size(-1))
    return (feature1 - feature2).pow(2).sum(1)

def batch_E_distance(feature1,feature2):
    batch_size = feature1.size(0)
    distances = []
    for i in range(batch_size):
        distances.append(E_distance(feature1[i], feature2[i]))
    # print(type(distances))
    return torch.cat(distances)



def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix
