from dotenv import dotenv_values
from pathlib import Path
import requests
import uuid
import json
from joblib import memory, Memory
from ps_funks import juteUtils as jut
from iso_language_codes import language_dictionary
from lingua import LanguageDetectorBuilder
from deep_translator import GoogleTranslator

_dicc_lang_name_2_iso = {v["Name"].upper(): k for k, v in language_dictionary().items()}
_dicc_lang_iso_2_name = {k: v["Name"].upper() for k, v in language_dictionary().items()}


# monkey patch this internal method
# to keep it from making new caches for each kernel
memory._build_func_identifier = lambda func: func.__name__
cachedir = Path.home() / "memory_cacher"
cachedir.mkdir(exist_ok=True)
memory = Memory(location=cachedir, verbose=0)


_d_repo = Path(__file__).resolve().parents[1]

def translate_auto_2_en(text, input_lang="auto"):
    """translates text with automatic language detection
    Args:
        text (_type_): _description_
        input_lang (str, optional): If the input language is known, you can specify it here. Note that it reduces the amount of characters that needs processing by 1/2. Defaults to "auto".

    Returns:
        text_in_en (str): translated text
    """
    # if language detection is applied, applied use the full text or enough of the text
    texts = jut.split_text_logical(text, 1500)
    for i, t in enumerate(texts):
        translation = translate_to_en(t, input_lang)
        texts[i] = translation
    text_in_en = ' '.join(texts)
    return text_in_en


def translate_to_en(text, input_lang=None, verbose=False):
    """translates to english with first trying to use google translate and then microsoft_translate (in case it fails)

    Args:
        text (str): text to translate
        input_lang (str, optional): iso2 language code of input text. Defaults to None.

    Returns:
        text (str): translated text
    """
    if input_lang == 'en':
        return text
    text = _translate_to_en(text, input_lang=input_lang, verbose=verbose)
    return text


@memory.cache
def _translate_to_en(text, input_lang, verbose=False):
    """helper function to cache the input, for description see translate_to_en

    Args:
        text (str): text to translate
        input_lang (str): iso code of input lang

    Returns:
        text (str): translated text
    """
    if verbose:
        print(f'about to translate (lang={input_lang}, len={len(text)}) by google')
    text = translate_to_en_by_google(text, input_lang=input_lang)
    # try:
    #     if verbose:
    #         print(f'about to translate (lang={input_lang}, len={len(text)}) by google')
    #     text = translate_to_en_by_google(text, input_lang=input_lang)
    # except:
    #     if verbose:
    #         print(f'\n about to translate (lang={input_lang}, len={len(text)}) by ms_api (google_failed)')
    #     body = [{"text": text}]
    #     text = (body, input_lang=input_lang)
    #     translation = free_translate_api(body, 'en', input_lang, f_out=f_out)
    #     text = translation[0]["translations"][0]["text"]
    return text


def translate_to_en_by_google(text, input_lang='auto'):
    """only translate if lang is not Language.ENGLISH and
    splits text in characters of maximum 1500 and applies translation to them

    Args:
        text (string): text to translate
        lang (Language): a lingua.Language object, e.g. Language.ENGLISH
    """
    if input_lang == 'en':
        return text
    translator = GoogleTranslator(source=input_lang, target="en")
    texts = jut.split_text_logical(text, 1500)
    for i, t in enumerate(texts):
        texts[i] = translator.translate(t)
    text = ' '.join(texts)
    return text


def detect_language(text, return_name=False):
    """automatic language detection (usefull to save characters, since auto detection doubles the amount of characters during translation)

    Args:
        text (str): text to translate
        return_name (bool, optional): if the language name shall be returned instead of iso2 code. Defaults to False.

    Returns:
        str: translated text
    """
    # if the text is too short, the language detection is not reliable enough
    if len(text.split(' ')) < 500:
        print('text too short to detect language by "detect_language"')
        return 'auto'
    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    lang_name = detector.detect_language_of(text)
    if lang_name:
        lang_name = lang_name.name
    if return_name:
        return lang_name
    if lang_name not in _dicc_lang_name_2_iso.keys():
        lang_iso = 'auto'
    else:
        lang_iso = _dicc_lang_name_2_iso[lang_name]
    return lang_iso


def free_translate_api(body, from_lang, to_lang, f_out=None, character_reduction=True):
    """
    uses the free translate api from microsoft
    the response is a list of dictionaries in the same order as the input list of texts
    Example:
        body = [
            {'text': 'This is the first entry.'},
            {'text': 'This is the second entry.'},
            ]
        f_out = Path.cwd() / 'test0.json'
        free_translate_api(body, 'en', 'de', f_out)
    """
    body = body.copy()  # to not change the original text (in case they used references)
    if character_reduction:
        for i in range(len(body)):
            body[i]["text"] = jut.squeeze_text(body[i]["text"])

    # Add your key and endpoint
    env = dotenv_values(_d_repo / ".env")
    key = env["key"]
    endpoint = env["endpoint"]
    location = env["location"]

    path = "/translate"
    constructed_url = endpoint + path

    params = {"api-version": "3.0", "from": from_lang, "to": to_lang}
    if from_lang == "auto":
        params.pop("from")

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": location,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    if f_out is not None:
        with open(f_out, "w") as f:
            json.dump(response, f, ensure_ascii=False, indent=4, separators=(",", ": "))
    return response
