import numpy as np

def assert_np_shape(a, shape: Union[List, Tuple]) -> bool:
    """
    assert_np_shape helper function to check the shapes of numpy arrays and vectors
    before operating on them using inbuilt function

    This function checks the all the shape of a numpy array
    each value in a.shape is checked against each value in shape
    value of elements in shape can be
        int: exact shape to be checked in a
        str: "n" value in this dimension is not checked

    :param a: numpy array to be checked
    :type a: np.array
    :param shape: iterable of shapes to be checked.
    :type shape: Union[List, Tuple]
    :return: whther shape of array is same as shape argument
    :rtype: bool
    """
    if len(a.shape) != len(shape):
        return False

    for dim in zip(a.shape, shape):
        if isinstance(dim[1], int) is False:
            continue
        if dim[0] != dim[1]:
            return False

    return True

def calculate_distance(x, x_test):
    """
    calculate_distance calculate the euclidaean distances for any point with respect
    to the training data

    for training data of shape (n,2) this function returns a vector (n,1) of distance
    wrt to each point.
    first dimension of x axis second of y axis

    :param x: training data points, shape (n,2)
    :type x: np.array
    :param x_test: new point wrt to which all distances need to be calculated, shape (1, 2)
    :type x_test: np.array
    :return: array of distances of every point in training set wrt to tesst point. shape (n,1)
    :rtype: np.array
    """

    assert assert_np_shape(x, ["n", 2])
    assert assert_np_shape(x_test, [1, 2])

    x_test = np.broadcast_to(x_test, x.shape)

    distance = (x - x_test) ** 2

    distance = np.sqrt(np.sum(distance, axis=1)).reshape((x.shape[0], 1))

    assert assert_np_shape(distance, [x.shape[0], 1])
    return distance


def sort_first_k(distances, y, k: int = 3):
    """
    sort_first_k sort the first k smallest elements in a distances array of shape (n,1)

    sort only the first k to reduce time complexity.
    np.argpartion is used
    np.argpartition(a, k-1) guarentees that the first k entries are the k-smallest entries in the
    whole array. They might not be in correct order but for KNN it doesn't matter. where k = {1, 2, 3, 4}

    :param distances: array of distances, shape(n,1)
    :type distances: np.array
    :param y: array of labels, shape (n,1)
    :type y: np.array
    :param k: number of smallest distances to sort, defaults to 3
    :type k: int, optional
    :return: Tuple of 2 arrays distances and labels of shapes (n,1) and (n,1)
    :rtype: Tuple[np.array, np.array]
    """

    assert assert_np_shape(y, ["n", 1])
    assert assert_np_shape(distances, ["n", 1])

    indices = np.argpartition(distances[:, 0], kth=k - 1)

    distances = distances[indices]
    y = y[indices]

    return distances, y



def knn_label(x, y, x_test, k: int = 3) -> int:
    """
    knn_label decide label based on k nearest neighbours

    _extended_summary_

    :param x: position labels as a (n,2) array
    :type x: np.array
    :param y: data labels as a (n, 1) array
    :type y: np.array
    :param x_test: labels for new points shape (1, 2)
    :type x_test: np.array
    :param k: parameter for looking the KNN algorithm, defaults to 3
    :type k: int, optional
    :return: scalar label corresponding to the new point
    :rtype: int
    """

    distances = calculate_distance(x=x, x_test=x_test)

    distances, y2 = sort_first_k(distances=np.copy(distances), y=np.copy(y))

    count_1 = np.sum(y2[:k, 0])
    count_0 = k - count_1

    if count_0 > count_1:
        return 0
    else:
        return 1


knn_label_vec = np.vectorize(knn_label)