# GreedyFeatureSelection

Perform greedy feature selection using either Shannon entropy or Rényi min-entropy, exploiting the power of C++ Python
bindings.
Adaptation of this [link](https://gitlab.com/marcoromane.gitlab.public/minentropyfeatureselection) using Python bindings instead
of pure Python. This is supposed to give better performance in not parallel execution. (For C++ based parallel execution check this [link](https://github.com/marcoromanelli-github/ParallelGreedyFeatureSelection)).

### Getting started
As first thing download the repo and move into the corresponding folder.
Let us create the .so and .cpp binding file by calling
```console
foo$bar python setup.py build_ext --inplace
```
Now let us put the generated cpython .so file in the py_code folder to run the script test_0.py. 

#### Todo
- [x] Basic implementation binding C++ and Python
- [ ] Extend to C++ parallel code
- [ ] Heavy testing
