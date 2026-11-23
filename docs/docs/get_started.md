# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep your system in order.

## Installing with `pip` (recommended)

The following command installs the latest version of the library:

```shell
pip install astrotech
```

To make the package as slim as possible, only the packages required to use `astrotech` are installed.
It is possible to install all the optional dependencies by running the following command:

```shell
pip install 'astrotech[all]'
```

This command also installed NumPy and PyTorch.
It is also possible to install the optional packages manually or to select the packages to install.
In the following example, only NumPy is installed:

```shell
pip install astrotech numpy
```

## Installing from source

To install `astrotech` from source, you can follow the steps below. First, you will need to
install [`uv`](https://docs.astral.sh/uv/). `uv` is used to manage the Python environment and to
install the dependencies. If `uv` is already installed on your machine, you can skip this step.
There are several ways to install `uv` so you can use the one that you prefer. You can check the
`uv` installation by running the following command:

```shell
uv --version
```

Then, you can clone the git repository:

```shell
git clone git@github.com:durandtibo/astrotech.git
```

The tasks of the project are defined with [`invoke`](https://www.pyinvoke.org/) and
[`invoke-tasklib`](https://github.com/durandtibo/invoke-tasklib). You can install them and create
a virtual environment with all the dependencies with the following command:

```shell
make setup-venv
```

This command creates a Python virtual environment in the `.venv` folder and installs the package
and its dependencies in it. When the virtual environment is created, you can activate it with the
following command:

```shell
source .venv/bin/activate
```

You can update the virtual environment and the dependencies at any time with the following
command:

```shell
inv env.update
inv env.install
```

Finally, you can test the installation with the following command:

```shell
inv test.unit
```

You can list all the available tasks with `inv --list`.
