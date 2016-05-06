import arff
import math

__author__ = 'SinLapis'


class Point():
    def __init__(self, arts, cla=''):
        self.arts = arts
        self.cla = cla


def read_train():
    file = open('iris.2D.train.arff')
    train_src = arff.load(file)['data']
    file.close()
    train_data = []
    for p_src in train_src:
        p = Point(p_src[0:2], p_src[2])
        train_data.append(p)
    return train_data


def read_test():
    file = open('iris.2D.test.arff')
    test_src = arff.load(file)['data']
    file.close()
    test_data = []
    for p_src in test_src:
        p = Point(p_src[0:2])
        test_data.append(p)
    return test_data


def count_distance(pos1, pos2):
    add = 0.0
    i = 0
    for art1 in pos1:
        art2 = pos2[i]
        add += (art1 - art2) ** 2
        i += 1
    return math.sqrt(add)


def knn(k):
    train_data = read_train()
    test_data = read_test()
    for test_point in test_data:
        distances = []
        point_cla = {}
        for train_point in train_data:
            distance = count_distance(test_point.arts, train_point.arts)
            distances.append(distance)
        sort_distances = []
        for d in distances:
            sort_distances.append(d)
        sort_distances.sort()
        max_distance = sort_distances[k - 1]
        i = 0
        for distance in distances:
            if distance <= max_distance:
                if train_data[i].cla in point_cla:
                    point_cla[train_data[i].cla] += 1
                else:
                    point_cla[train_data[i].cla] = 1
            i += 1
        max_vote = 0
        for key in point_cla:
            if point_cla[key] > max_vote:
                max_vote = point_cla[key]
                test_point.cla = key
    return test_data


if __name__ == '__main__':
    k = int(input('Put in the k:'))
    test_data = knn(k)
    for test_point in test_data:
        print(test_point.arts, test_point.cla)
