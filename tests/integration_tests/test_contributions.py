import unittest
import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from transparency.decomposition.contributions import compute_contributions

class TestContributions(unittest.TestCase):

    def setUp(self):
        x, y = load_iris(return_X_y=True)
        self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(x, y, random_state=1)
    
    def check(self, s, b, x_test):
        assert len(s) == 3
        assert len(b) == 3
        for i in range(2):
            assert s[i].shape == x_test.shape
            assert 0 <= s[i].min().min() <= 1
            assert 0 <= s[i].max().max() <= 1
            assert 0 <= b[i] <= 1

    def test_compute_contributions_1(self):
        model = RandomForestClassifier(n_estimators=3)
        model.fit(self.x_train, self.y_train)
        explainer = shap.TreeExplainer(model)
        s, b = compute_contributions(self.x_test, explainer)
        self.check(s, b, self.x_test)

    def test_compute_contributions_2(self):
        model = GradientBoostingClassifier(n_estimators=3)
        model.fit(self.x_train, self.y_train)
        explainer = shap.TreeExplainer(model)
        s, b = compute_contributions(self.x_test, explainer)
        self.check(s, b, self.x_test)
    
    def test_compute_contributions_3(self):
        model = LogisticRegression()
        model.fit(self.x_train, self.y_train)
        explainer = shap.LinearExplainer(model, self.x_train)
        s, b = compute_contributions(self.x_test, explainer)
        self.check(s, b, self.x_test)
    
    def test_compute_contributions_4(self):
        model = SVC()
        model.fit(self.x_train, self.y_train)
        explainer = shap.KernelExplainer(model, self.x_train)
        s, b = compute_contributions(self.x_test, explainer)
        self.check(s, b, self.x_test)
    
    def test_compute_contributions_5(self):
        model = MLPClassifier(hidden_layer_sizes=(3, ))
        model.fit(self.x_train, self.y_train)
        explainer = shap.DeepExplainer(model, self.x_train)
        s, b = compute_contributions(self.x_test, explainer)
        self.check(s, b, self.x_test)

    # TODO : idem for lightgbm, XGBoost, keras, pytorch
