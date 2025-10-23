import numpy as np
from pathlib import Path
import re
import pandas as pd
import polars as pl
import geopandas as gpd
import geodatasets
from tabulate import tabulate
from shapely.geometry import MultiPolygon, Polygon
from shapely.affinity import translate
from functools import partial
from joblib import Memory, memory


# monkey patch this internal method
# to keep it from making new caches for each kernel
memory._build_func_identifier = lambda func: func.__name__
cachedir = Path.home() / "memory_cacher"
cachedir.mkdir(exist_ok=True)
memory = Memory(location=cachedir, verbose=0)


def normalize_text(text):
    """normalize text by removing special characters and lowering all letters"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return text


def split_text_logical(text, threshold, breaker=None):
    breaker_forced = breaker is not None
    text = squeeze_text(text)  # must not contain '\n' since this is used to split
    if breaker is None:
        breaker = "。" if "。" in text else ". "  # in chinese a circle is used instead of a dot
    text = text_addLineBreaks(text, threshold, breaker, splitter="\n\n")
    texts = text.split("\n\n")
    if not breaker_forced:
        lens = [len(t) for t in texts]
        if np.max(lens) > 1.5 * threshold:
            print(f'split-error breaker="{breaker}", using breaker=" "', end="\r")
            text = " ".join(texts)
            texts = split_text_logical(text, breaker=" ", threshold=threshold)
    return texts


def squeeze_text(text, replace_newline=False):
    """remove all multiple newlines and multiple spaces
        - important for the free translate api (counts per character and also spaces count as characters)
        - note that replace of newline is problematic for languages as Japanese

    Args:
        text (str): text
        replace_newline (bool, optional): if newline shall be replaced. Defaults to False.

    Returns:
        text (str): squeezed text
    """
    text = re.sub("\n+", "\n ", text)
    if replace_newline:
        text = re.sub("\n", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip(' ')


def gpd_get_world(print_neglected=False):
    """
    loads and from geopandas (gpd) the dataset "naturalearth_lowres"
    and substitues missing iso-alpha3 codes
    NOTE:
        - Somaliland is internationally only recognized as independent country by Taiwan
            but is autonomously governed
    """

    # world = gpd.read_file(gpd.datasets.get_path("naturalearth.land"))
    world = gpd.read_file(_d_data / 'geoBoundariesCGAZ_ADM0.zip')
    # rename:
    world = world.rename(columns={'shapeGroup': 'iso_a3', 'shapeName': 'country_name'})
    # exclude regions without a proper iso_a3 code, like:
    # ['Abyei', 'Aksai Chin', 'CH-IN', 'Demchok', 'Dragonja', 'Dramana-Shakatoe', 'Falkland Islands (UK)', 'Gaza Strip', 'Kalapani', 'Isla Brasilera',
    #  'Siachen-Saltoro', 'Koualou', 'Liancourt Rocks', "No Man's Land", 'Paracel Is', 'Sanafir & Tiran Is.', 'Senkakus', 'Spratly Is', 'West Bank']
    mask_alpha = world.iso_a3.apply(lambda x: x.isalpha())
    if print_neglected:
        print(world.loc[~mask_alpha].values)
    world = world.loc[mask_alpha]
    return world


@memory.cache
def gpd_get_world_regions():
    """
    same as gpd_get_world but not on country level but world_region level
    """
    df_countryInfo = get_countryInfo()
    world = gpd_get_world()
    # cs_arptInfo = ['country_name', 'region_name', 'continent_name', 'region_id', 'continent_id']
    world = world.set_index("iso_a3").join(
            df_countryInfo.drop(columns=["continent_name", "country_name", "iso_alpha2"])
            )
    world = world.loc[~(world.region_name.isna())]
    df_world_r = (
        world.dissolve(by=(cby := "region_name"))
        .drop(columns=["country_name"])
        .reset_index()
    )
    return df_world_r


@memory.cache
def gpd_get_world_regions_stacked():
    world = gpd_get_world_regions().query('region_name != "Antarctica"')
    # compute centroids of americas and the rest as rough translates
    center_americas_x = -84.106148
    center_rest_x = 69.573751
    y_min = -55.91544342 # min-y value of americas
    y_max2 = 81.86164093 # max-y value or rest of the world
    # translate operations
    x_off1 = - center_americas_x + 20
    y_off1 = 0
    x_off2 = - center_rest_x
    y_off2 = - y_max2 + y_min - 5
    # now prepare the translate function
    translate_conditionally_part = partial(translate_conditionally, x_limit=-10,
                                           x_off1=x_off1, y_off1=y_off1,
                                           x_off2=x_off2, y_off2=y_off2)
    # Apply the function to each geometry in the GeoDataFrame
    world['geometry'] = world['geometry'].apply(translate_conditionally_part)
    return world


def translate_conditionally(geometry, x_limit=0, x_off1=0, y_off1=0, x_off2=0, y_off2=0):
    ''' Function to translate each polygon based on its max x-bound '''
    translate1 = partial(translate, xoff=x_off1, yoff=y_off1)
    translate2 = partial(translate, xoff=x_off2, yoff=y_off2)
    if geometry.is_empty:
        return geometry
    if isinstance(geometry, Polygon):
        # Handle single Polygon
        bounds = geometry.bounds
        max_x = bounds[2]  # Extract max x-bound
        if max_x < x_limit:
            return translate1(geometry)
        else:
            return translate2(geometry)
    elif isinstance(geometry, MultiPolygon):
        # Handle MultiPolygon
        translated_polygons = []
        for polygon in geometry.geoms:
            bounds = polygon.bounds
            max_x = bounds[2]  # Extract max x-bound
            polygon = translate1(polygon) if max_x < x_limit else translate2(polygon)
            translated_polygons.append(polygon)
        return MultiPolygon(translated_polygons)
    else:
        # Return the geometry as is if it's not a Polygon or MultiPolygon
        return geometry



def text_addLineBreaks(string, N, breaker=None, splitter=None):
    """
    adds line breaks to a string at the first "breaker" after every N characters
    the "breaker" is per default a space, i.e. it will not break words
    """
    breaker = " " if breaker is None else breaker
    splitter = "\n" if splitter is None else splitter
    splits = string.split(breaker)
    newName = splits[0]
    for sp in splits[1:]:
        newName += breaker + sp
        if len(newName.split(splitter)[-1]) > N:
            newName += splitter
    if newName[-len(splitter) :] == splitter:
        newName = newName[: -len(splitter)]
    return newName


def get_joined_tabula(dfs, same_length=True):
    """
    INPUT:
        dfs list of pandas.DataFrames
        same_length bool
            if True: shortens all tabulas to the shortest one
            else: sort the tabulas according to their length (longest first)
    NOTE:
        for saving without clipping use bbox_inches='tight':
        f.savefig('name.pdf', bbox_inches='tight')
    """
    if same_length:
        return get_joined_tabula_same_length(dfs)
    else:
        return get_joined_tabula_sorted(dfs)


def get_tabula(df):
    tabula = tabulate(df, headers="keys", tablefmt="psql")
    return tabula


def squezze_repeated_char(text, char):
    out = re.sub(f"({char})+", char, text)
    return out


def df_column_snake_name(df, sep="_"):
    polars = type(df) == pl.DataFrame
    # rename columns
    renamer = {
        c: c.replace(".", sep)
        .replace("…", sep)
        .replace("(", sep)
        .replace(")", "")
        .replace(" ", sep)
        .replace("%", f"in{sep}percent")
        .lower()
        .strip(sep)
        for c in df.columns
    }
    renamer = {k: squezze_repeated_char(v, sep) for k, v in renamer.items()}
    if polars:
        df = df.rename(renamer)
    else:
        df = df.rename(columns=renamer)
    return df


def df_among_duplicate_replace_with_max(df, duplicate_criteria, column_to_replace, add_duplicate_group_column=False):
    """replace the value of column_to_replace with the maximum value among duplicates

    Args:
        df (pandas.DataFrame): DataFrame with duplicates
        duplicate_criteria (str): column to identify duplicates
        column_to_replace (str): column to replace the value

    Example:
        ```python
        df = pd.DataFrame({'last_name': ['Smith', 'Smith', 'Bells', 'Bells'],
                           'first_name': ['Sam', 'S', 'John', 'J.']})
        df_among_duplicate_replace_with_max(df, 'last_name', 'first_name')
    """
    # replace the value of column_to_replace with the maximum value among duplicates
    is_duplicate = df[duplicate_criteria].duplicated(keep=False)
    unique_dups = df.loc[is_duplicate, duplicate_criteria].unique()
    c_dup = 'duplicate_group'
    df[c_dup] = None
    for i, dup in enumerate(unique_dups):
        is_this_dup = df[duplicate_criteria] == dup
        df.loc[is_this_dup, [column_to_replace, c_dup]] = [df.loc[is_this_dup,
                                                                  column_to_replace].dropna().max(),
                                                           i]
    if not add_duplicate_group_column:
        df.drop(columns=[c_dup], inplace=True)



def setDefault(x, val):
    if x is None:
        x = val
    return x


def get_countryInfo():
    """
    contains basic country information, as population, GDP, etc.
    - the 14 'regions' values in 'countries_with_missing_regions.csv' were set by hand based on proximity
    """
    df = pd.read_csv(_d_data / 'country_data.csv', index_col=0)
    # iso2 for namibia gets always messed up "NA" with NaN
    df.loc['NAM', 'iso_alpha2'] = "NA"
    return df


_d_data = Path(__file__).parent.resolve() / "data"


def get_joined_tabula_same_length(dfs):
    """
    INPUT:
        dfs list of pandas.DataFrames
    """
    sep_tabu = "   "
    minlen = len(dfs[0])
    lens = np.unique([len(df) for df in dfs])
    if len(lens) > 1:
        minlen = np.min(lens)
    stri0 = get_tabula(dfs[0].iloc[:minlen])
    stri0 = stri0.split("\n")
    for df in dfs[1:]:
        stri1 = get_tabula(df.iloc[:minlen])
        stri1 = stri1.split("\n")
        stri0 = [s0 + sep_tabu + s1 for s0, s1 in zip(stri0, stri1)]
    stri0 = "\n".join(stri0)
    return stri0


def get_joined_tabula_sorted(dfs):
    """
    longest tabula, second longest ....
    """
    sep_tabu = "   "
    dfs = sorted(dfs, key=len, reverse=True)
    stri0 = get_tabula(dfs[0])
    stri0 = stri0.split("\n")
    for df in dfs[1:]:
        stri1 = get_tabula(df)
        stri1 = stri1.split("\n")
        len1 = len(stri1)
        stri00 = [s0 + sep_tabu + s1 for s0, s1 in zip(stri0[:len1], stri1)]
        stri0 = stri00 + stri0[len1:]
    stri0 = "\n".join(stri0)
    return stri0


def treeDict(dic, level=0, name="", maxKeys=5):
    """creates a tree view of the dictionary

    Args:
        dic (dict): dictionary of interest
        level (int, optional): treeDict is a recursive function and level is the counter of the recursion-level. Defaults to 0.
        name (str, optional): name to be displayed. Defaults to ''.
        maxKeys (int, optional): if the list of keys is longer than maxKeys --> only display first 5. Defaults to 5.
    """
    string0 = "  " * level + f"-{name}: "
    if type(dic) == dict:
        ks = list(dic.keys())
        if len(ks) > maxKeys:
            string0 += f"Showing only {maxKeys} of total {len(ks)} keys: ({ks})"
            ks = ks[:maxKeys]
        print(string0, ks)
        for k in ks:
            treeDict(dic[k], level=level + 1, name=k, maxKeys=maxKeys)
    elif hasattr(dic, "shape"):
        print(string0, "shape=", dic.shape)
    elif type(dic) in [list, tuple]:
        print(string0, f"type={type(dic)}, length={len(dic)}, {name}[0]={dic[0]}")
    else:
        print(string0, dic)


def df_xbin_yquantile(df, x, y, bins=50, quantiles=None):
    '''
    creates x-bins that have all the same amount of data and computes in these bins
    the [0.25, 0.5, 0.75] y-quantiles
    and
    [x_mean, x_median, y_mean]

    INPUT:
        df pd.DataFrame
        x str || float || int
            dataframe column along which the data is binned
            in bins with the same amount of data
        y str || float || int
            dataframe column of which the quantiles of each bin are computed
    OUTPUT:
        df pd.DataFrame
            index= x-quantiles
            columns= [0.25, 0.5, 0.75, x_mean, x_median, y_mean]
    '''
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    df = df.copy()
    df['x_qs'] = pd.qcut(df[x], bins, duplicates='drop')
    df = df.groupby('x_qs', observed=False)[y].quantile(quantiles)\
            .to_frame().reset_index()\
            .pivot(index='x_qs', columns='level_1', values=y)\
            .join(df.groupby('x_qs', observed=False)[x].mean().rename('x_mean'))\
            .join(df.groupby('x_qs', observed=False)[x].median().rename('x_median'))\
            .join(df.groupby('x_qs', observed=False)[y].mean().rename('y_mean'))
    return df


def nanptp(arr):
    """Calculate the range of values in an array, ignoring NaN values.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        float: Range of values in the array, ignoring NaN values.
    """
    valid_values = arr[~np.isnan(arr)]
    return np.ptp(valid_values)


def find_blocks_low(dat, mind, minNr, minBlock, return_indices=False, lowEqual=False):
    '''
    find start- end end-time of blocks where
    dat<mind for at least "minNr" agents
    INPUT:
        dat.shape(time, N) OR (time)
            datability that ID is correct for "N" agents OR 1 agent
        minNr float
            # of agents for which the creterion must hold simultaneously
        minBlock int
            minium size of block-length
    '''
    there = np.where(dat < mind)
    if lowEqual:
        there = np.where(dat <= mind)
    if len(dat.shape) > 1:  # dat.shape(time, N)
        Blowdats = np.zeros(dat.shape)
        Blowdats[there] = 1
        there = np.where(np.sum(Blowdats, axis=1) >= minNr)[0]
    else:                   # dat.shape(time)
        there = there[0]
    if return_indices:
        return there
    blocks = get_blocks(there, minBlock)
    return blocks


def get_blocks(there, minsize):
    '''
    return blocks(continously increasing values) in there
    and return the start and end-value of the blocks
    example:
            in: there=[1, 2, 3, 10, 11, 12, 13, 14, 22, 25,26]
            out: blocks=[[1,3], [10, 14], [22, 22], [25, 26]]

        INPUT:
            there.shape(time)
        OUTPUT:
            blocks.shape(blocks, 2)
    '''
    if len(there) == 0:
        return []
    differs = np.diff(there)
    borders = np.where(differs>1)[0]
    sblocks = np.ones((len(borders) + 1, 1), dtype='int')  # start points of blocks
    eblocks = np.ones((len(borders) + 1, 1), dtype='int')  # end points of blocks
    sblocks[0] = there[0]  # first start point of block
    eblocks[-1] = there[-1]  # last end point of block
    for i in range(len(borders)):
        eblocks[i] = there[borders[i]]
        sblocks[i+1] = there[borders[i] + 1]
    blocks = np.hstack((sblocks, eblocks))
    blocklen = np.diff(blocks) + 1
    longenough = np.where(blocklen >= minsize)[0]
    # ppk
    print(f'minsize = {minsize}')
    blocks = blocks[longenough]
    return blocks
