import os

from biosignalspicklereader import BioSignalsReader
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import svm
import numpy as np

if __name__ == '__main__':
    chest_sensors = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
    sensor = "ECG"
    BSR = BioSignalsReader(sensor=sensor)
    x_mat, y_mat = BSR.prepare_feature_matrix()
    rs = ShuffleSplit(n_splits=10, test_size=0.3)
    combined_text = ""
    i = 1
    scores = []

    for train_index, test_index in rs.split(x_mat):
        x_train = x_mat[train_index]
        x_test = x_mat[test_index]
        y_train = y_mat[train_index]
        y_test = y_mat[test_index]

        clf = svm.SVC(kernel='linear', C=1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        scores.append(acc_score)
        combined_text += f"Test number {i}\n"
        combined_text += f"Accuracy score: {acc_score}\n"
        combined_text += f"Confusion matrix: \n {conf_matrix}\n" \
                         f"--------------------------------\n"
        i += 1

    combined_text += f"/**********************************/\n" \
                     f"Average accuracy score: {np.average(scores)}\n"

    if not os.path.isdir("results"):
        os.mkdir("results")

    f = open(f"results/{sensor}_results.txt", "w")
    f.write(combined_text)
    f.close()
