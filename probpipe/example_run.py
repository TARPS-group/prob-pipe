from prefect import flow, task
import probpipe
from probpipe.core.module import Module


mod = Module()

# Set default inputs
mod.set_input(x=10, y=20)

# Define a run function with validation and type hints
@mod.run_func()
def add(x: int, y: int) -> int:
    return x + y

# Define a validate method on the 'add' function for custom validation
def add_validate(x, y):
    if x < 0 or y < 0:
        raise ValueError("x and y must be non-negative")

# Attach the validate function to 'add'
add.validate = add_validate

# Try running with valid inputs (uses default y=20)
print("Result with x=5 (y default 20):", mod.run(name="add", x=5))

# Try running with invalid input (negative number) - should raise ValueError
try:
    print("Attempt with negative x=-1:", mod.run(name="add", x=-1))
except Exception as e:
    print(f"Caught validation error as expected: {e}")

# Try running with wrong type - string instead of int - should raise TypeError
try:
    print("Attempt with x='foo':", mod.run(name="add", x="foo"))
except Exception as e:
    print(f"Caught type error as expected: {e}")