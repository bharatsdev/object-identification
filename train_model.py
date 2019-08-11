from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.model_selection import cross_val_score
from scipy.stats.stats import pearsonr

face_embeddings = "data/output/embeddings.pickle"
print("[INFO] : Loading faces name from embedding...")
face_embedd_data = pickle.loads(open(face_embeddings, "rb").read())
faceNames = face_embedd_data['names']
print("[INFO] : Fetching faces name from embedding...\n{}".format(faceNames))
print("[INFO] : Label encoding for faces Names..")
le = LabelEncoder()
facesEncoding = le.fit_transform(faceNames)
print(facesEncoding)

print("[INFO] : Training ML model on generated Face Embedding....")
X = face_embedd_data["embeddings"]
Y = facesEncoding
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=41)

svm = SVC(random_state=41)

from sklearn.model_selection import GridSearchCV

param_grid = {'kernel': ('linear', 'rbf'), 'C': (1, 0.25, 0.5, 0.75), 'gamma': (1, 2, 3, 'auto'),
              'decision_function_shape': ('ovo', 'ovr'), 'shrinking': (True, False)}
grid = GridSearchCV(svm, param_grid=param_grid, verbose=3)

# svclassifier = SVC( verbose=True, random_state=41)
grid.fit(trainX, trainY)
print(grid.best_params_)
print(grid.best_estimator_)

y_pred = grid.predict(testX)
from sklearn.metrics import classification_report, confusion_matrix

print("accuracy:" + str(np.average(cross_val_score(grid, trainX, Y, scoring='accuracy'))))
print("f1:" + str(np.average(cross_val_score(grid, trainX, Y, scoring='f1'))))

print(confusion_matrix(testY, y_pred))
print(classification_report(testX, y_pred, target_names=faceNames))

print("[INFO] : Serialize the SVC classifier model...")
f = open("data/output/facerecognizer.pickle", "wb")
f.write(pickle.dumps(svm))
f.close()

print("[INFO] : Serialize label encoder to desk...")

f = open("data/output/labelencoder.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
