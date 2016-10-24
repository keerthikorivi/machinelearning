from sklearn.ensemble import RandomForestClassifier
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import numpy as np

#color references for the matrix:http://matplotlib.org/examples/color/colormaps_reference.html
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.GnBu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_roc_curve(X_test_label,test_predicted,class_names):
    X_test_label_binary = label_binarize(X_test_label, classes=class_names)
    test_predicted_binary = label_binarize(test_predicted, classes=class_names)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(X_test_label_binary, test_predicted_binary)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    #plt.imshow(cmap=plt.cm.GnBu)
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='ROC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def readFile(inputFile,input,label):
    with open(inputFile,"r") as source:
        rdr= csv.reader( source )
        for r in rdr:
            x=[]
            x.append(r[0])
            x.append(r[1])
            x.append(r[2])
            input.append(x)
            label.append(r[3])


train_input=[[]]
train_label=[]
test_input=[[]]
test_label=[]
class_names=np.array(['low_poverty','high_poverty'])
print('reading train file')
readFile("C://Keerthi//backup//Masters//CIS890//dataset//final//train.csv",train_input,train_label)
print('reading test file')
readFile("C://Keerthi//backup//Masters//CIS890//dataset//final//test.csv",test_input,test_label)
print('creating classifier...')

#creating classifier
clf=RandomForestClassifier(n_estimators=1000,criterion="gini",max_features=0.9,random_state=42)
train_input.remove([])
X_train=np.array(train_input)
X_train_label=np.array(train_label)
test_input.remove([])
#fit the training data into the classifier
clf.fit(X_train,X_train_label)
X_test=np.array(test_input)
X_test_label=np.array(test_label)
test_predicted=np.array(clf.predict((X_test)))


#Accuracy
print('Accuracy : ',accuracy_score(X_test_label,test_predicted))

precision_recall_f1_score=precision_recall_fscore_support(X_test_label, test_predicted, average='macro')
#Precision
print('Precision: ' , precision_recall_f1_score[0])
#Recall
print('Recall: ' , precision_recall_f1_score[1])
#F1 score
print('F1 Score: ' , precision_recall_f1_score[2])


# Compute confusion matrix
cnf_matrix = confusion_matrix(X_test_label, test_predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plot_roc_curve(X_test_label,test_predicted,class_names)

