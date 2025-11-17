# from .module import Module, InputSpec
from types import SimpleNamespace, MappingProxyType
from contextlib import contextmanager
import inspect
from typing import Callable, get_type_hints

from probpipe import Module, Distribution 
from typing import Any, Dict, List


"""
Node acts like an edge in a directed graph.
It transforms data from one vertex to another using a function.
"""




class Node:
    """
    Node represents a single computation step.
    It takes an input key from the workflow state,
    applies a function, and stores the result under an output key.
    """
    def __init__(self, input_key, output_key, func):
        self.input_key = input_key
        self.output_key = output_key
        self.func = func

    def run(self, state):
        value_in = state[self.input_key]
        value_out = self.func(value_in)
        state[self.output_key] = value_out
        return value_out

    def __repr__(self):
        return f"Node({self.input_key} -> {self.output_key}, func={self.func.__name__})"




class Workflow:
    """
    Workflow executes nodes sequentially.
    Each node reads from and writes to the shared state dict.
    """
    def __init__(self):
        self.nodes = []
        self.state = {}

    def add_node(self, node: Node):
        self.nodes.append(node)

    def set_input(self, **kwargs):
        self.state.update(kwargs)

    def run(self):
        for node in self.nodes:
            node.run(self.state)
        return self.state

    def __repr__(self):
        lines = ["Workflow:"]
        for n in self.nodes:
            lines.append(f"  {n}")
        return "\n".join(lines)





def module1(x): 
    print("Module1"); return x + 1

def module2(x): 
    print("Module2"); return x * 2

def module3(x): 
    print("Module3"); return {"mu": x, "sigma": 1}

def module4(post): 
    print("Module4"); post["sigma"] += 0.5; return post



def node1_transform(data):
    return module2(module1(data))

def node2_posterior(tdata):
    return module4(module3(tdata))





wf = Workflow()

# Node1 = data transformation
node1 = Node("data", "transformed", node1_transform)

# Node2 = posterior computation
node2 = Node("transformed", "posterior", node2_posterior)

wf.add_node(node1)
wf.add_node(node2)

wf.set_input(data=5)

print(wf)
print("\n--- Running Workflow ---")
result = wf.run()
print("\nFinal Result:", result)
