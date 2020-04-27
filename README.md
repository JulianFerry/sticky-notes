# mnist-keras-gcp

**Note: To enable Tensorflow's IntelliSense for VS Code, do the following ([source](https://github.com/tensorflow/tensorflow/issues/32982#issuecomment-545414061)):**

- Find your tensorflow_core installation (the python package). This is e.g. in ~/.local/lib/python3.6/site-packages, any other site-packages folder you might use (e.g. from virtualenv, pyenv,...)
- Create a folder to use for IDE navigation, e.g. ~/.local/virtual-site-packages
- Create a symlink in that folder called tensorflow to your tensorflow_core package (mind the name difference, this is intentional!)
- Add the path created in the 2nd step to python.autoComplete.extraPaths in VSC (use the full path, i.e. replace your username)

For example from `.venv/` for Python 3.7:

``` bash
mkdir $PWD/virtual-site-packages
ln -s $PWD/lib/python3.7/site-packages/tensorflow_core $PWD/virtual-site-packages/tensorflow
```

Then add the following to `settings.json`:

``` json
"python.autoComplete.extraPaths": [
        "${workspaceFolder}/.venv/virtual-site-packages"
],
```
