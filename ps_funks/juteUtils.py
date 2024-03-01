import numpy as np
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd
from tabulate import tabulate


def split_text_logical(text, threshold, breaker=None):
    breaker_forced = breaker is not None
    text = squeeze_text(text) # must not contain '\n' since this is used to split
    if breaker is None:
        breaker = "。" if "。" in text else '. ' # in chinese a circle is used instead of a dot
    text = text_addLineBreaks(text, threshold, breaker, splitter='\n\n')
    texts = text.split('\n\n')
    if not breaker_forced:
        lens = [len(t) for t in texts]
        if np.max(lens) > 1.5 * threshold:
            print(f'split-error breaker="{breaker}", using breaker=" "', end='\r')
            text = ' '.join(texts)
            texts = split_text_logical(text, breaker=' ', threshold=threshold)
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
    return text


def gpd_get_world(print_unrecognized=False):
    """
    loads and from geopandas (gpd) the dataset "naturalearth_lowres"
    and substitues missing iso-alpha3 codes
    NOTE:
        - Somaliland is internationally only recognized as independent country by Taiwan
            but is autonomously governed
    """

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    # add iso3 codes that before were -99
    world.loc[world.name.str.contains("France"), "iso_a3"] = "FRA"
    world.loc[world.name.str.contains("Norway"), "iso_a3"] = "NOR"
    world.loc[world.name.str.contains("Kosovo"), "iso_a3"] = "XKX"
    world.loc[world.name.str.contains("Somaliland"), ["iso_a3", "name"]] = "SOM", "Somalia"
    # there are still 'N. Cyprus' that are listed as -99
    if print_unrecognized:
        print(world.loc[world.iso_a3 == "-99"])
    return world


def gpd_get_world_regions():
    """
    same as gpd_get_world but not on country level but world_region level
    """

    df_countryInfo = get_countryInfo()
    world = gpd_get_world()
    # cs_arptInfo = ['country_name', 'region_name', 'continent_name', 'region_id', 'continent_id']
    world = world.set_index("iso_a3").join(df_countryInfo.drop(columns=["continent", "name"]))
    world = world.loc[~(world.region_name.isna())]
    df_world_r = (
        world.dissolve(by=(cby := "region_name"))
        .drop(columns=[(cd := "pop_est"), "name", "country_name"])
        .join(world.groupby(cby)[cd].sum())
        .reset_index()
    )
    return df_world_r


def text_addLineBreaks(string, N, breaker=None, splitter=None):
    """
    adds line breaks to a string at the first "breaker" after every N characters
    the "breaker" is per default a space, i.e. it will not break words
    """
    breaker = ' ' if breaker is None else breaker
    splitter = '\n' if splitter is None else splitter
    splits = string.split(breaker)
    newName = splits[0]
    for sp in splits[1:]:
        newName += breaker + sp
        if len(newName.split(splitter)[-1]) > N:
            newName += splitter
    if newName[-len(splitter):] == splitter:
        newName = newName[:-len(splitter)]
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
    out = re.sub(f"\{char}\{char}+", char, text)
    return out


def df_column_snake_name(df, sep="_"):
    # rename columns
    renamer = {
        c: c.replace(".", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace(" ", "_")
        .replace("%", "")
        .lower()
        for c in df.columns
    }
    df = df.rename(columns=renamer)
    renamer = {c: squezze_repeated_char(c, "_") for c in df.columns}
    df = df.rename(columns=renamer)
    return df


def setDefault(x, val):
    if x is None:
        x = val
    return x


def get_countryInfo():
    """
    contains basic country information, as population, GDP, etc.
    """
    df_countryInfo = pd.read_csv(_d_data / "country_data.csv").set_index("iso_alpha3")
    return df_countryInfo


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
