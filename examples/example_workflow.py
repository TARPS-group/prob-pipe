from typing import Optional, Callable
from probpipe import WorkflowBuilder, Module 
from prefect import flow, task
import numpy as np
from numpy.typing import NDArray

# Simple Example for WorkflowBuilder usage

# Example subclasses representing components/dependencies
class DataModule(Module):
    def __init__(self):
        super().__init__()
        # Add any data-specific initialization here

class LikelihoodModule(Module):
    def __init__(self):
        super().__init__()
        # Add likelihood-specific initialization here

class PriorModule(Module):
    def __init__(self):
        super().__init__()
        # Add prior-specific initialization here


with WorkflowBuilder(name="MyBayesWorkflow") as builder:
    # Register modules (dependencies)
    builder.register_modules({
        "data": DataModule(),
        "likelihood": LikelihoodModule(),
        "prior": PriorModule(),
    })

    # Register run function to perform the workflow
    @builder.register_run
    def run(self, x):
        print(f"Running workflow with input: {x}")
        print(f"Dependencies available: {list(self.dependencies.keys())}")

        # Access dependency modules here:
        data_mod = self.dependencies["data"]
        lik_mod = self.dependencies["likelihood"]
        prior_mod = self.dependencies["prior"]


        result = x * 10  
        print(f"Computed result: {result}")
        return result

# Outside the with block, the workflow is built automatically

output = builder.run_workflow(5)  # Run the workflow with input argument
print(f"Workflow output: {output}")

