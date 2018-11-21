import preProcess
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
from sklearn.neural_network import MLPClassifier
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    def showConfusionMatrix(self):
        encoder_file = open("models/encoder.pk", "rb")
        encoder = pickle.load(encoder_file)
        clf = joblib.load("models/clf_RandomForestClassifier.joblib")
        encoder_file.close()
        plt.figure()
        cm = confusion_matrix(self.p.test_labels, clf.predict(self.p.test_features))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        tick_marks = np.arange(len(np.unique(self.p.test_labels)))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.xticks(tick_marks, np.unique(self.p.test_labels), rotation=90)
        plt.yticks(tick_marks, np.unique(self.p.test_labels))
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def showAccuracy(self):
        encoder_file = open("models/encoder.pk", "rb")
        encoder = pickle.load(encoder_file)
        clf = joblib.load("models/clf_RandomForestClassifier.joblib")
        encoder_file.close()

        words2feature = encoder.transform(["233"])
        w=clf.predict_proba(words2feature)
        print(w)
        print(max(max(w)))


if __name__ == '__main__':
    t = Train()
    #t.showConfusionMatrix()
    t.showAccuracy()
