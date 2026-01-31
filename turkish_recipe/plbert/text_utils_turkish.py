# Turkish PL-BERT Text Utils
# Expanded phoneme vocabulary for Turkish Wikipedia dataset

import os
import string

_pad = "$"

# Original punctuation from PL-BERT
_punctuation = ';:,.!?¡¿—…"«»"" '

# Additional characters found in Turkish Wikipedia phonemes
_turkish_extra = '\t\n!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \xa0'

# Merge punctuation (deduplicate)
_all_punctuation = ''.join(sorted(set(_punctuation + _turkish_extra)))

# Letters
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Original IPA symbols
_letters_ipa_original = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Turkish-specific IPA additions from dataset analysis
_turkish_ipa_extra = "ɫɭɯɲɳɕɖɜɟɻʂʈʐʑ̃ⁿ"

# Combine all IPA (deduplicate)
_letters_ipa = ''.join(sorted(set(_letters_ipa_original + _turkish_ipa_extra)))

# Include digit '1' which appears in data
_digits = '01234567890'

# Export all symbols (matching original structure)
symbols = [_pad] + list(_all_punctuation) + list(_letters) + list(_letters_ipa) + list(_digits)

# Remove duplicates while preserving order
seen = set()
symbols_unique = []
for s in symbols:
    if s not in seen:
        seen.add(s)
        symbols_unique.append(s)
symbols = symbols_unique

letters = list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len(symbols)):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(f"Turkish TextCleaner initialized with {len(dicts)} symbols")
        
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                # Unknown token - map to 'U' or pad
                if 'U' in self.word_index_dictionary:
                    indexes.append(self.word_index_dictionary['U'])
                else:
                    indexes.append(0)  # pad token
        return indexes

# Export vocab size for config
VOCAB_SIZE = len(symbols)

if __name__ == "__main__":
    print(f"Total symbols: {VOCAB_SIZE}")
    print(f"Symbols: {''.join(symbols[:50])}...")
    tc = TextCleaner()
    test = "merhaba dünya"
    print(f"Test encoding '{test}': {tc(test)}")
