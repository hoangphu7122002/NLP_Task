import json
from tkinter import N
import unicodedata
import re

class VniErrorCorrector(object):
    fi = open('telex/complex_telex_fault.json', encoding='utf-8')
    complex_telex = json.load(fi)
    eng_file = open('teencode/eng_dic.json', 'r')
    eng_dic = json.load(eng_file)
    
    def __init__(self):
        self.build_character_regexs()
        self.build_accent_regexs()
    
    def fix_vni_sentence(self, sentence):
        sentence = unicodedata.normalize('NFC', sentence)
        # print(sentence)
        words = [self.fix_vni_word(word) for word in sentence.split()]
        sentence = ' '.join(words)
        # sentence = unicodedata.normalize('NFC', sentence)
        return sentence

    def fix_vni_word(self, word):
        try:
            if self.eng_dic[word] == 1: 
                return word
        except:
            word = word.lower()
            word = unicodedata.normalize('NFKD', word)
            
            for accent, keystroke in self.accent_vni_errors.items():
                word = re.sub(accent, keystroke, word)
            
            reorder_word = word
            i = 1
            n = len(word)
            while (i < n):
                a = reorder_word[i]
                if a in self.additional_keystrokes + ['w'] and (reorder_word[i - 1] + a != 'tr'):
                    reorder_word = reorder_word[:i] + reorder_word[i+1:] + a
                    n = n - 1
                else:
                    i = i + 1
            word = reorder_word
    
            word = unicodedata.normalize('NFC', word)
            
            for key, value in self.char_vni_errors.items():
                word = re.sub(key, value, word)
    
            ## 'trưong' -> 'trương'
            word = re.sub('ưo', 'ươ', word)
            
            for key, value in self.char_vni_errors.items():
                word = re.sub(key, value, word)
        
            for key, value in self.accent_vni_errors.items():
                word = re.sub(key, value, word)
            
            return word
        
    def build_character_regexs(self):
        chars = ['ă', 'â', 'ư', 'ô', 'ơ', 'ê','đ']
        additional_keystrokes = ['8','6','7', '6', '7', '6','9']
        
        char_vni_errors = dict()
        
        for i, c in enumerate(chars):
          parts = unicodedata.normalize('NFKD', c) #to denormalize sign
          base_c = parts[0]
          keystroke = additional_keystrokes[i]
          pattern = f'{base_c}(.*){keystroke}'
          char_vni_errors[pattern] = c + '\\1'
        char_vni_errors['d(.*)9'] = 'đ\\1'
        #a
        char_vni_errors['á(.*)7'] = 'ắ\\1'
        char_vni_errors['à(.*)7'] = 'ằ\\1'
        char_vni_errors['ả(.*)7'] = 'ẳ\\1'
        char_vni_errors['ã(.*)7'] = 'ẵ\\1'
        char_vni_errors['ạ(.*)7'] = 'ặ\\1'
        
        char_vni_errors['á(.*)6'] = 'ấ\\1'
        char_vni_errors['à(.*)6'] = 'ầ\\1'
        char_vni_errors['ả(.*)6'] = 'ẩ\\1'
        char_vni_errors['ã(.*)6'] = 'ẫ\\1'
        char_vni_errors['ạ(.*)6'] = 'ậ\\1'
        #o       
        char_vni_errors['ó(.*)7'] = 'ớ\\1'
        char_vni_errors['ò(.*)7'] = 'ờ\\1'
        char_vni_errors['ỏ(.*)7'] = 'ở\\1'
        char_vni_errors['õ(.*)7'] = 'ỡ\\1'
        char_vni_errors['ọ(.*)7'] = 'ợ\\1'
        
        char_vni_errors['ó(.*)6'] = 'ố\\1'
        char_vni_errors['ò(.*)6'] = 'ồ\\1'
        char_vni_errors['ỏ(.*)6'] = 'ổ\\1'
        char_vni_errors['õ(.*)6'] = 'ỗ\\1'
        char_vni_errors['ọ(.*)6'] = 'ộ\\1'
        #e
        char_vni_errors['é(.*)6'] = 'ế\\1'
        char_vni_errors['è(.*)6'] = 'ề\\1'
        char_vni_errors['ẻ(.*)6'] = 'ể\\1'
        char_vni_errors['ẽ(.*)6'] = 'ễ\\1'
        char_vni_errors['ẹ(.*)6'] = 'ệ\\1'
        #u
        char_vni_errors['ú(.*)7'] = 'ứ\\1'
        char_vni_errors['ù(.*)7'] = 'ừ\\1'
        char_vni_errors['ủ(.*)7'] = 'ử\\1'
        char_vni_errors['ũ(.*)7'] = 'ữ\\1'
        char_vni_errors['ụ(.*)7'] = 'ự\\1'
                
    
        self.char_vni_errors = char_vni_errors
    
    def build_accent_regexs(self):
        chars = ['a', 'u', 'o', 'e', 'i', 'y', 'ă', 'â', 'ư', 'ô', 'ơ', 'ê']
        accents = ['í', 'ỉ', 'ĩ', 'ì', 'ị']
        accents = [unicodedata.normalize('NFKD', a)[1] for a in accents]
        additional_keystrokes = ['1', '3', '4', '2', '5']
        
        self.additional_keystrokes = additional_keystrokes
        
        accent_vni_errors = dict()
    
        for c in chars:
          for i, a in enumerate(accents):
            text = ''.join([c, a])
            merged = unicodedata.normalize('NFC', text)
            
            keystroke = additional_keystrokes[i]
            pattern = f'{c}(.*){keystroke}'
            # print(pattern,merged)
            accent_vni_errors[pattern] = merged + '\\1'
    
        self.accent_vni_errors = accent_vni_errors

if __name__ == '__main__':
    a = VniErrorCorrector()
    fixed = a.fix_vni_sentence('đếch muốn thế')
    print(fixed)