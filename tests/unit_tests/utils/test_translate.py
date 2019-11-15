import unittest
from transparency.utils.translate import translate

class TestTranslate(unittest.TestCase):
    
    def test_translate(self):
        elements = ['X_1', 'X_2']
        mapping = {'X_1': 'âge', 'X_2': 'profession'}
        output = translate(elements, mapping)
        expected = ['âge', 'profession']
        self.assertListEqual(output, expected)
