# LSMemoryModel
A codebase that allows you to play with different online learning algorithms that model memory 

### How to install:
Run the following sequence of all commands
```
make virtulenv
source venv/bin/activate
make all
pre-commit install
```
This should create a virtual env called venv, install all dependencies,
install the package and activate the created virtual environment.

### How to use pre-commit
`pre-commit` will run automatically as a hook so every commit will need
to adhere to the linter. Use the following work-stream. 
Assume there are files `file.txt` and `script.py`. Then the workflow is
```
git add file.txt
git add script.py
pre-commit
... [fix all of the things that can't be automatically fixed ] ...
git add file.txt
git add script.txt
git commit -m "message"
```

