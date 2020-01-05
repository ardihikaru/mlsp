from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv
from hw5.libs.algo.knn_theano import KNNTheano
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # dataset = Dataset(train_data=1000, test_data=100)
    # dataset = Dataset(train_data=40, test_data=10)
    dataset = Dataset(train_data=80, test_data=20)
    # dataset = Dataset()
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    train_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_scores = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    exec_times = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    best_test_score = 0
    stored_accuracy = [0, 0] # [<acc>, <k-th>

    # Define number of iteration (K)
    K = 50 # min. value = 10
    ks = int_to_tuple(K)  # used to plot the results

    # Start KNN: sklearn
    print("Evaluating sklearn KNN")
    for i in range(9, K):
        k = i+1
        print("\nk =", k)
        knn = KNNTheano(k)
        knn.fit(X_train, Y_train)

        t0 = datetime.now()
        train_score = knn.score(X_train, Y_train)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))

        t0 = datetime.now()
        test_score = knn.score(X_test, Y_test)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))
        exec_times.append((datetime.now() - t0).total_seconds())

        # update best accuracy (for comparison purpose later)
        if test_score > best_test_score:
            best_test_score = test_score
            stored_accuracy = [test_score, k]

        # np.savetxt(fname=csv_path + 'exec-knn-theano.csv', X=str("dsd"), delimiter=',', fmt='%d')

    fig = plt.figure()
    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()
    fig.savefig('hw5/results/result-knn-theano.png', dpi=fig.dpi)
    save_to_csv('exec-knn-theano.csv', exec_times)
    save_to_csv('knn-test-scores-theano.csv', test_scores)
    save_to_csv('knn-best-acc-theano.csv', stored_accuracy)




