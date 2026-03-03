# Development install

```shell
uv sync
```

## Add dependencies

Regular or development dependencies are added as following
```shell
uv add numpy
uv add pytest --dev
```

## Run tests

```shell
uv run pytest
```

## Formatting and checking 

```shell
ruff format
ruff check --fix
```

... but `ruff format` is known to break the code for Python 3.8.
So, not recommended until the unit tests are covering 100% of the code. 

## Bump version number 

```shell
bump-my-version bump patch
```

## Upgrading the dependencies

```shell
uv sync -U
```

## List the versions of the installed dependencies

```shell
uv tree --depth 1
```


