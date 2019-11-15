import unittest
from unittest.mock import Mock
from transparency.decomposition.contributions import compute_contributions

class TestContributions(unittest.TestCase):
    
    def test_compute_contributions_1(self):
        x = Mock()
        explainer = Mock()
        output = compute_contributions(x, explainer)
        explainer.shap_values.assert_called()
        explainer.expected_value.assert_called()
        assert len(output) == 2
    
    def test_compute_contributions_2(self):
        x = Mock()
        explainer = Mock()
        preprocessing = Mock()
        output = compute_contributions(x, explainer, preprocessing)
        preprocessing.transform.assert_called()
        explainer.shap_values.assert_called()
        explainer.expected_value.assert_called()
        assert len(output) == 2
