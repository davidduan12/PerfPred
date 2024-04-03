from sklearn.impute import KNNImputer
import sys
import os
current_working_directory = os.getcwd()
print("Current working directory:", current_working_directory)
# Add the parent directory to the system path
sys.path.append(os.path.abspath('..'))
from utils import *
import matplotlib.pyplot as plt
import time


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    matrix_t = matrix.T
    imputer = KNNImputer(n_neighbors=k)
    imputed_matrix = imputer.fit_transform(matrix_t)
    imputed_matrix = imputed_matrix.T
    
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    val_accuracies_user_based = []
    val_accuracies_item_based = []
    
    # user-based part
    for k in k_values:
        print(f"Running user-based collaborative filtering with k = {k}")
        a = time.time()
        val_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        b = time.time()
        print("Running time: {}".format(b - a))
        val_accuracies_user_based.append(val_acc)
        
    best_k = k_values[val_accuracies_user_based.index(max(val_accuracies_user_based))]
    print(f"Best k value: {best_k}")
    print(f"Validation set accuracy with k = {best_k}: {max(val_accuracies_user_based)}")
    
    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print(f"Test accuracy with k = {best_k}: {test_acc}")
    
    print("Now start item based part")
    
    # item based part
    for k in k_values:
        print(f"Running item-based collaborative filtering with k = {k}")
        a = time.time()
        val_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        b = time.time()
        val_accuracies_item_based.append(val_acc)
        print("running time: ", b-a)
    
    best_k_item_based = k_values[val_accuracies_item_based.index(max(val_accuracies_item_based))]
    print(f"Best k value for item-based: {best_k_item_based}")
    print(f"Validation set accuracy with k = {best_k_item_based}: {max(val_accuracies_item_based)}")
    
    test_acc_item_based = knn_impute_by_item(sparse_matrix, test_data, best_k_item_based)
    print(f"Test accuracy with k = {best_k_item_based} for item-based: {test_acc_item_based}")
    
    # plot
    plt.plot(k_values, val_accuracies_user_based, marker='o', label='User-based')
    plt.plot(k_values, val_accuracies_item_based, marker='x', label='Item-based')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Collaborative Filtering: Validation Accuracy vs. k')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
