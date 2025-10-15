import unittest
import numpy as np
from probpipe.core.module import Module


class TestModule(unittest.TestCase):

    def setUp(self):
        # Define test subclasses for dependencies
        class DataModule(Module):
            def __repr__(self):
                return "<DataModule>"
        class ModelModule(Module):
            def __repr__(self):
                return "<ModelModule>"
        self.DataModule = DataModule
        self.ModelModule = ModelModule

    def test_add_dependency_and_duplicate(self):
        data = self.DataModule()
        mod = Module()
        mod.add_dependency('data', data)
        self.assertIn('data', mod.dependencies)
        self.assertEqual(mod.dependencies['data'], data)

        with self.assertRaises(ValueError):
            mod.add_dependency('data', data)  # duplicate

    def test_instantiate_defaults(self):
        mod = Module()
        mod.instantiate(a=1, b=2)
        self.assertEqual(mod._defaults['a'], 1)
        self.assertEqual(mod._defaults['b'], 2)

    def test_register_run_fun_and_duplicate(self):
        mod = Module()

        @mod.decorate_run_fun
        def fn(x):
            return x

        self.assertIn('fn', mod._run_funcs)

        with self.assertRaises(RuntimeError):
            mod.decorate_run_fun(fn)  # duplicate by same name

    def test_automatic_input_detection(self):
        mod = Module()
        dep = self.DataModule()
        mod.add_dependency('data', dep)

        @mod.decorate_run_fun
        def run(data, param1: int = 5, param2='foo'): ## data: self.DataModule
            return param1, param2

        self.assertIn('param1', mod.inputs)
        self.assertIn('param2', mod.inputs)
        self.assertNotIn('data', mod.inputs)  # dependency not an input

        self.assertEqual(mod.inputs['param1']['default'], 5)
        self.assertEqual(mod.inputs['param2']['default'], 'foo')

    def test_run_function_execution(self):
        data = self.DataModule()
        model = self.ModelModule()
        mod = Module(data=data, model=model)
        mod.instantiate(epochs=10)

        @mod.decorate_run_fun
        def train(data, model, epochs: int = 1, lr: float=0.01):
            ## data: self.DataModule, model: self.ModelModule
            return (str(data), str(model), epochs, lr)

        result = mod.run()
        self.assertEqual(result, ('<DataModule>', '<ModelModule>', 10, 0.01))

        result = mod.run(epochs=5, lr=0.1)
        self.assertEqual(result, ('<DataModule>', '<ModelModule>', 5, 0.1))

    def test_run_missing_input_raises(self):
        mod = Module()

        @mod.decorate_run_fun
        def func(x):
            return x

        with self.assertRaises(ValueError):
            mod.run()  # x missing

    def test_run_no_run_funcs_raises(self):
        mod = Module()
        with self.assertRaises(RuntimeError):
            mod.run()

    def test_repr_and_str(self):
        mod = Module()
        mod.add_dependency('dep1', self.DataModule())
        mod.instantiate(param=1)

        @mod.decorate_run_fun
        def f(x=1):
            pass

        # repr should include dependency, input, run_funcs keys
        r = repr(mod)
        self.assertIn('dep1', r)
        self.assertIn('param', r)  # input name
        self.assertIn('f', r)  # run function name

        # str should include nicely formatted info
        s = str(mod)
        self.assertIn('Dependencies:', s)
        self.assertIn('Inputs:', s)
        self.assertIn('Run Functions:', s)