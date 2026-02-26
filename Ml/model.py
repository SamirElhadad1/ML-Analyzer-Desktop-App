from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_model(X, y, task, algorithm):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if task == "classification":
        if algorithm == "KNN":
            model = KNeighborsClassifier()
        elif algorithm == "SVM":
            model = SVC()
        else:
            model = DecisionTreeClassifier()
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test
