from ps_funks import translate as tra
from pathlib import Path

_d_test_data = Path(__file__).resolve().parents[0] / 'data'

def test_translate_to_en_by_google():
    text = 'Ich würde gerne eine Pizza bestellen.'
    text_en = tra.translate_to_en_by_google(text, input_lang='auto')
    print(text)
    print('---->')
    print(text_en)
    assert text_en == 'I would like to order a pizza.'

# def test_free_translate_api():
#     print('Attention: if the API key is not set in the ".env" file, this test will fail.',
#           '(the .env file is not in the repository for security reasons)')
#     body = [
#         {'text': 'This is the first entry.'},
#         {'text': 'This is the second entry.'},
#         ]
#     _d_test_data.mkdir(exist_ok=True)
#     f_out = _d_test_data / 'test0.json'
#     translations = tra.free_translate_api(body, 'en', 'de', f_out=f_out)
#     print(body)
#     print('--->')
#     print(translations)
#     assert translations[0]['translations'][0]['text'] == 'Dies ist der erste Eintrag.'
#     assert translations[1]['translations'][0]['text'] == 'Dies ist der zweite Eintrag.'

def test_translate_auto_2_en():
    print('auto translation to english')
    text = 'Ich würde gerne Pizza bestellen.'
    text_in_en = tra.translate_auto_2_en(text)
    print(text)
    print('--->')
    print(text_in_en)
    assert text_in_en == 'I would like to order pizza.'
