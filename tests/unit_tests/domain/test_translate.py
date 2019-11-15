import unittest
from transparency.domain.translate import translate_names

class TestTranslateNames(unittest.TestCase):
    
    def test_translate_name(self):
        old_names = ['X_1', 'X_2']
        features_dict = {'X_1': 'âge', 'X_2': 'profession'}
        output = translate_names(old_names, features_dict)
        expected = ['âge', 'profession']
        self.assertListEqual(output, expected)