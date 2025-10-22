# PsFunks


**function collection for plotting and general purpose functions**

## Motivation

In order to not redefine general purpose functions in different repositories, they should be collected here.
This includes plotting, translation, text processing and model management functions/classes.

---

## Install
0. optional create conda environment:
    *  `conda env create -f env.yml`
    *  `conda activate ps_funks`
1. normal pip-install: `pip install .`
OR
1. **developer mode** pip-install: `pip install -e . --user`

Note: if there are dependencies problem use the pinned environment [./env_pinned.yml](./env_pinned.yml).

## Dependencies

* plotting related:
    * **matplotlib**: used for plotting
    * **seaborn**: to use swarmplot
    * **cycler**: to create a color cycler object (color-blind friendly)
* to access the microsof-translator api in `translate.py`
    * **requests** 
    * **uuid** 
* **numpy**: for numerical computations
* **pathlib**: for a portable way to navigate file-systems
* **pandas**: to use DataFrame objects 
* **geopandas**: for map representations
* **tabulate**: to create tables
* **python-dotenv**: to access user-specific environment settings in a local `.env` file
* **scikit-learn**: to perform some board-analysis (split data in test and training, ...)
* **deep-translator**: to perform text translations

## FILES
short explanation of all files in the repo:

* `ps_funks/`: folder contains [code, modules]
    * `__init__`: init file
    * `data/`: folder contains data files for the code
        * `geoBoundariesCGAZ_ADM0.zip`: ADM0 (Countries) shapefile dowloaded from [https://www.geoboundaries.org/globalDownloads.html](https://www.geoboundaries.org/globalDownloads.html)
            * it is needed for the world map with countries
        * `country_data_simple.csv`: country dat with [`iso_alpha2`, `iso_alpha3`, `country_name`, `region_name`, `continent_name`]
        * `countries_with_missing_regions.csv`: same as above but region-data was added by hand
    * `hotPlot.py`: all functions related to plotting
    * `juteUtils.py`: all functions unrelated to plotting
    * `board_operations.py`: contains the class BoardData to handle multiple boards with the same sk-learn settings 
    * `translate.py`: basic functions to translate text, depends on api-key from microsoft translator
        * ATTENTION: only works with `.env` file in root-directory of the repo (see Secrets section)
* `tests/`: folder contains test files 
    * `test_with_pytest.py`: example file to test tesing via e.g. pytest
    * `test_hotPlot.py`: incomplete tests for `hotPlot.py` module 
    * `test_juteUtils.py`: incomplete test for `juteUtils.py` module 
    * `test_translate.py`: tests for `translate.py` module
        * ATTENTION: only works with `.env` file in root-directory of the repo (see Secrets section)
    * `test_board_operations.py`: tests for `board_operations.py` module
        * ATTENTION: depends on `test_translate.py` -> again on `.env` file
    * `data/`: folder that contains data to run tests
        * `test_board_1.csv`: labeled data from a board, just needed to test the loading
        * `test_board_2.csv`: see `test_board_1.csv`
* `env.yml`: YAML environment file than specifies dependencies from other packages
* `LICENSE`: LICENSE File
* `pyproject.toml`: contains the various settings (black, ruff, ...) for the project
* `README.md`: the readme
* `setup.py`: needed for install via pip

## Usage

### plotting with hotPlot 

* first load the module
* to use costumized matplotlib settings, put at the beginning of your notebook/script
```python
from ps_funks import hotPlot as hp
hp.setRcParams()
```
* in order to save figures in multiple formats define your `savefig_multi`
```python
from Pathlib import Path
from functools import partial 

d_figs = Path.cwd().parent / "figures"
d_figs.mkdir(exist_ok=True)
savefig_multi = partial(hp.savefig_multiformat, d_figs)  # formats=["png", "pdf", "svg"]
```
* now when saving a figure it creates in `path_to_your_repo/figures` the folders "png", "pdf", "svg" and save the figure in the respective format
    * if other formats are needed, define them in the block above via `formats` option
```python
f, ax = plt.subplots(1)
# do some plotting here
savefig_multi(f, 'fig_name')
```
* very usefull for multipanel figures is `hp.subplots(N)` which is a wrapper of `plt.subplots()`
    * with the options `n_row` and `n_col` you either define how many rows or columns the grid shall have
    * it will return a flattened array of axes-objects (and takes care of too many created due to grid constraints)
```python
N = 5
f, axs = hp.subplots(N, n_col=2)
for i in range(N):
    ax = axs[i]
    # do some plotting on ax
``` 
* very frequent used functions:
```python
# to put in the upper left corner ABC-labels run:
hp.abc_plotLabels([0.01, 0.9], axs, fontsize=12)
# if the x-axes has dates use (makes pandas-style xaxis)
hp.nice_dates(ax, monthstep=3)
```
* other useful functions:
```python
# to create a histogram with logarithmic bins:
hp.hist_logx(ax, values, 20)
# to make the marker of the legend have the same size:
hp.legend_handle_same_size(ax.legend(), size=30)
```

### quick with juteUtils

```python
from ps_funks import juteUtils as jut 
```
* to load a slightly corrected geopandas world and a region-version of it run:
```python
world = jut.gpd_get_world()
world_r = jut.gpd_get_world_regions()
```
* you can also load country information as population, GDP(pc), etc by `df_ci = jut.get_countryInfo()`
* to get a string representation of a pandas DataFrame
```python
txt = jut.get_tabula(df.head(30))
# or to show 2 tables on the same line use
txt = jut.get_joined_tabula([df1, df2])
```
* there are some text processing commands
```python
# the code below replaces all repetitions of " " with a single " "
jut.squezze_repeated_char(txt, " ")
# modifies the column names of a pandas DataFrame to snake-style
# "GDP (pc)" --> "gdp_pc" 
df = jut.df_column_snake_name(df)
```

### analyzing Text data with board_operations

This module allows the analysis of labeled articles and assumes that the labelling was done via [label-studio](https://labelstud.io/).
* an example can be found in [./tests/data/test_board_1.csv](./tests/data/test_board_1.csv)
    * the labeled data has columns "sentiment" with entries ['Noise', 'Maybe Noise', 'Not Noise']
```python
from ps_funks import board_operations as bo
```
* create a BoardData object and fill it with different boards and create a model you want to test
    * it binarizes the "sentiment" column ('Noise' -> 0, else 1), translates the text
```python
# the random_state ensures reproducability
board_data = bo.BoardData(test_size=0.33, random_state=42)
# now 
board_data.add_board('/path/to/your/studio_labeled_data1.csv', 'name1')
board_data.add_board('/path/to/your/studio_labeled_data2.csv', 'name2')
classifier = YourModelFunction()    # created for example via sklearn.pipeline.Pipeline
```
* now train and test your model
```python
train_board = 'name1'
# training
x, y, dat_name = board_data.get_train_xy(train_board)
classifier.fit(x, y)
x, y, dat_name = board_data.get_test_xy(train_board)
df_test = bo.test_prediction_df(classifier, x, y, multi_index=train_board)

# testing
for board_name in set(board_data.board_names) - set([train_board]):
    x, y, mlf_data = board_data.get_strat_xy(board_name)
    df_test_ = bo.test_prediction_df(classifier, x, y, multi_index=board_name)
    df_test = pd.concat([df_test, df_test_], axis=1)
```
* now your `df_test` contains the predictions results for your trained model on all boards

## License
* for the microsoft-translator used in `translate.py` an api key is needed
    * when creating you have to accept the terms of use

## Secrets
In order for the translation api from microsoft to work you need a secret `.env` file that is not contained in this repo for security reasons. An example `.env` file is:

```bash
key=<secret_key>
location=<your_location>
endpoint=https://api.cognitive.microsofttranslator.com/
```

## ToDos:

* module 'translate.py' is partly obsolete
    * the python package **deep-translator** also supports microsoft-translator, replace the call with this one
