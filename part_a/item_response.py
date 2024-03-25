from utils import *
import matplotlib.pyplot as plt

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    # Loop over all data points
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]

        # Calculate the probability of the correct answer
        prob_correct = sigmoid(theta[user_id] - beta[question_id])

        # Update the log likelihood for the correct and incorrect answers
        if is_correct:
            log_lklihood += np.log(prob_correct)
        else:
            log_lklihood += np.log(1 - prob_correct)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def gradient_theta_beta(data, theta, beta):
    """ Compute the gradients for theta and beta according to the given formula. """
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)

    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]

        z = theta[user_id] - beta[question_id]
        prob_correct = sigmoid(z)

        grad_theta[user_id] += is_correct - prob_correct
        grad_beta[question_id] += prob_correct - is_correct

    return grad_theta, grad_beta


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    grad_theta, grad_beta = gradient_theta_beta(data, theta, beta)
    theta += lr * grad_theta
    beta += lr * grad_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(set(data['user_id'])))
    beta = np.zeros(len(set(data['question_id'])))

    valid_lst = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        valid_lst.append(neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, valid_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def plot_log_likelihoods(train_ll):
    """
    Plot the log likelihoods of training data.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_ll, label='Training Log-likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.title('Training and Validation Log-likelihoods')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_probability_curves(theta, beta, questions):
    """
    Plot the 3 probability curves.
    """
    theta_range = np.linspace(-3, 3, 100)
    plt.figure(figsize=(10, 6))

    for q in questions:
        beta_q = beta[q]
        p_correct = 1 / (1 + np.exp(-(theta_range - beta_q)))
        plt.plot(theta_range, p_correct, label=f'Question {q + 1}')

    plt.xlabel('Ability ($\\theta$)')
    plt.ylabel('Probability of Correct Response')
    plt.title('Probability of Correct Response vs. Ability')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # learning_rates = np.arange(0.0001, 0.11, 0.01)
    # iteration_counts = np.arange(100, 1000, 100)
    learning_rates = [0.001]
    iteration_counts = [100]

    # num_users = len(set(train_data['user_id']))
    # num_questions = len(set(train_data['question_id']))
    # # theta = np.zeros(num_users)
    # # beta = np.zeros(num_questions)

    tuning_results = {}
    val_acc_lst = None
    val_lst = None

    for lr in learning_rates:
        for iterations in iteration_counts:
            theta_tuned, beta_tuned, val_acc_lst, val_lst = irt(train_data, val_data, lr, iterations)
            val_acc = evaluate(val_data, theta_tuned, beta_tuned)
            tuning_results[(lr, iterations)] = val_acc
            print(f"LR: {lr}, Iterations: {iterations}, Validation Accuracy: {val_acc}")

        # Find the best hyperparameters
    best_params = max(tuning_results, key=tuning_results.get)
    best_lr, best_iterations = best_params
    print(f"Best LR: {best_lr}, Best Iterations: {best_iterations}")

    theta_final, beta_final, _, _ = irt(train_data, val_data, best_lr, best_iterations)
    test_acc = evaluate(test_data, theta_final, beta_final)
    print(f"Final Test Accuracy: {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    # part (c)
    plot_log_likelihoods(val_lst)

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    questions_to_plot = [0, 1, 2]
    plot_probability_curves(theta_final, beta_final, questions_to_plot)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
