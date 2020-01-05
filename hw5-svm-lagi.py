from hw5.libs.common.dataset import Dataset
from hw5.libs.common.util import int_to_tuple, save_to_csv
from hw5.libs.algo.KDEClassifier import KDEClassifier
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # dataset = Dataset(train_data=1000, test_data=100)
    # dataset = Dataset(train_data=40, test_data=10)
    dataset = Dataset(train_data=80, test_data=20)
    # dataset = Dataset()
    X_train, Y_train, X_test, Y_test = dataset.get_dataset()
    train_scores = []
    test_scores = []
    # exec_times = [0]
    # best_test_score = 0
    # stored_accuracy = [0, 0] # [<acc>, <k-th>

    # Define value of bandwidth
    K = 50 # Must be positive and higher than 0 (ZERO)
    step_size = 0.05
    step_init = 0.05
    ks = int_to_tuple(K)  # used to plot the results
    # next:
    # 1. bikin array sesuai step size, contoh: ks = (0.05, 0.1, 0.15, 0.2, ...) sampai sejumlah K
    # 2. looping sejumlah K
    # 3. di setiap looping, lakukan perhitungan sesuai Cass KDE (fit - prob - acc)
    # 4. ending, kasih tau cara pencarian OPTIMAL Bandwidth, hitung estimasi untuk setiap class!!!! Lalu lakukan MEAN/AVG untuk dipakai sbg bw paling optimal.
    # 5. Running lagi dengan bw optimal. bandingkan hasilnya.

    # Start KDE: sklearn
    print("Evaluating sklearn KDE")
    for i in range(9, K):
        k = i+1
        print("\nk =", k)
        kde = KDEClassifier(k)
        kde.fit(X_train, Y_train)

        t0 = datetime.now()
        train_score = kde.score(X_train, Y_train)
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




