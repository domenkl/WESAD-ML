from biosignalspicklereader import BioSignalsReader
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import svm

if __name__ == '__main__':
    chest_sensors = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
    BSR = BioSignalsReader(sensor='ECG')
    x, y = BSR.prepare_train_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # rs = ShuffleSplit(n_splits=10, test_size=0.2)

    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
    # bsr.save_all_graphs(g_type='all')
    # bsr.draw_subject_graph('S3', show=False, save=True)
    # import numpy as np
    #
    # a = np.array([5, 10, 10, 10, 7, 3, 10])
    # b = np.array([1, 2, 3, 4, 5, 6, 7])
    #
    # print(np.asarray(a) == 10)
    # print(b[a == 10])
