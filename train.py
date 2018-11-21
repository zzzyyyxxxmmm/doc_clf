import preProcess
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
from sklearn.neural_network import MLPClassifier

names = ["SVM", "MLPClassifier", "GradientBoostingClassifier", "RandomForestClassifier"]

classifiers = [
    SVC(verbose=True),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, verbose=True),
    GradientBoostingClassifier(verbose=True),
    RandomForestClassifier(verbose=True)
]


class Train:

    def __init__(self):
        self.p = preProcess.PreProcess(100000, "dataset/shuffled-full-set-hashed.csv")
        self.p.run()

    def train(self, method, save_model=False):
        print("start training with " + method)
        index = names.index(method)
        if index == -1:
            print("wrong classifier")
            return
        clf = classifiers[index]
        clf.fit(self.p.train_features, self.p.train_labels)
        print("end train with " + method)
        print("Accuracy:", clf.score(self.p.test_features, self.p.test_labels))
        if (save_model):
            s = pickle.dumps(clf)
            model_path = 'models/clf_' + method + ".joblib"
            joblib.dump(clf, model_path)

    def train_with_svm(self, save_model=False):
        print("start training with SVM")
        clf_svm = SVC(verbose=True)
        clf_svm.fit(self.p.train_features, self.p.train_labels)
        print("Accuracy:", clf_svm.score(self.p.test_features, self.p.test_labels))
        print("end train with SVM")
        if (save_model):
            s = pickle.dumps(clf_svm)
            joblib.dump(clf_svm, 'models/clf_svm.joblib')


if __name__ == '__main__':
    t = Train()
    t.train("MLPClassifier", True)
