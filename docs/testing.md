# Testing

This document explains how to run and manage tests for the Autodidact project.

---

## Test Structure
All tests are located in the `tests/` directory. They are organized to match the structure of the main project modules.

Example:
```
tests/
    test_concepts.py
    test_core.py
    test_memory.py
    ...
```

---

## Running Tests

### Using `pytest`
The project uses `pytest` as the primary testing framework.

To run all tests:
```bash
pytest
```

To run tests for a specific file:
```bash
pytest tests/test_memory.py
```

To run a specific test function:
```bash
pytest tests/test_memory.py::test_graph_builder
```

---

## Test Coverage
To check test coverage, install `pytest-cov`:
```bash
pip install pytest-cov
```

Run tests with coverage:
```bash
pytest --cov=.
```

---

## Writing New Tests
When adding new features or fixing bugs:
1. Create a new test file in `tests/` if one doesn't exist for the module.
2. Use descriptive test function names.
3. Follow the Arrange-Act-Assert pattern:
    - **Arrange**: Set up the test data and environment.
    - **Act**: Execute the function or method under test.
    - **Assert**: Verify the results.

Example:
```python
def test_example_function():
    # Arrange
    input_data = [1, 2, 3]
    
    # Act
    result = example_function(input_data)
    
    # Assert
    assert result == [2, 3, 4]
```

---

## Continuous Integration
If using GitHub Actions or another CI tool, ensure that the test suite runs automatically on pull requests to maintain code quality.
