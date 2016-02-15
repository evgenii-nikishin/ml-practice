# -*- coding: utf-8 -*-

from my_functions import calc_entropy, calc_entropy2
import unittest
import numpy as np
import numpy.testing as npt

class TestMyFunctions(unittest.TestCase):
    
    #Эта функция вызывается перед каждым тестом
    def setUp(self):
        self.data1 = np.array([0, 1, 1])
        self.data2 = np.array([0, 0, 0])
        self.data3 = np.array([0, 1])
        self.data4 = np.arange(10)
    
    #Каждый тест реализуется функцией с именем, начинающимся на test
    def test_calc_entropy_not_array_input(self):
        self.assertRaises(TypeError, calc_entropy, [1, 2, 3])
        self.assertRaises(TypeError, calc_entropy, 'string')
        
    def test_calc_entropy_not_1d_array(self):
        self.assertRaises(TypeError, calc_entropy, np.array([1, 2, 3]).reshape(3,1))
        self.assertRaises(TypeError, calc_entropy, np.array([[1, 2, 3], [2, 3, 4]]))
        
    def test_calc_entropy_empty_input(self):
        self.assertRaises(TypeError, calc_entropy, np.array([]))

    def test_calc_entropy_typical_entry(self):
        self.assertAlmostEqual(calc_entropy(self.data1), -(2/3)*np.log2(2/3) - (1/3)*np.log2(1/3))
    
    def test_calc_entropy_one_label(self):
        self.assertAlmostEqual(calc_entropy(self.data2), 0)
        
    def test_calc_entropy_two_labels(self):
        self.assertAlmostEqual(calc_entropy(self.data3), 1)
        
    def test_calc_entropy_uniform_labels(self):
        npt.assert_allclose(calc_entropy(self.data4), np.log2(10)) #Эта функция проводит сравнение значений с учётом относительной и абсолютной погрешности
        
    def test_calc_entropy2(self):
        self.assertAlmostEqual(calc_entropy(self.data1), calc_entropy2(self.data1))
        self.assertAlmostEqual(calc_entropy(self.data2), calc_entropy2(self.data2))
        self.assertAlmostEqual(calc_entropy(self.data3), calc_entropy2(self.data3))
        self.assertAlmostEqual(calc_entropy(self.data4), calc_entropy2(self.data4))


if __name__ == "__main__":
    unittest.main()
    