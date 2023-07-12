import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ebooklib import epub
import inflect
import spacy
import nltk
import numpy as np
from scipy.io import wavfile

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import SAMPLE_RATE

import argparse
from html.parser import HTMLParser

# determine if a given string is a number, including decimals and commas
# str -> bool
def is_number(word):
    for char in word:
        if not (char.isdigit() or char == "." or char == ","):
            return False
    return True

# in a string, replace all numbers with words
# e.g. "I have 2 apples" -> "I have two apples"
# str -> str
def replace_numbers_with_words(text):
    months = ["January", "February", "March", 
            "April", "May", "June", 
            "July", "August", "September",
            "October", "November", "December"]
    
    p = inflect.engine()

    words = text.split()
    for i in range(len(words)):
        word = words[i]

        # remove punctuation from the beginning and end of the word
        post_punctuation = ""
        while len(word) > 0 and not word[-1].isalnum():
            post_punctuation += word[-1]
            word = word[:-1]

        post_punctuation = post_punctuation[::-1]

        if len(word) == 0:
            continue

        pre_punctuation = ""
        while not word[0].isalnum():
            pre_punctuation += word[0]
            word = word[1:]
        
        if word.isdigit(): # number with just digits
            if len(word) == 4: # year
                words[i] = ' '.join(p.number_to_words(word, group=2, wantlist=True))
            elif (len(word) == 1 or len(word) == 2) and i > 0 and words[i - 1] in months: # day of month
                words[i] = p.ordinal(p.number_to_words(word))
            else: # number
                words[i] = p.number_to_words(word)
        elif word.endswith("s") and word[:-1].isdigit(): # number with s
            number = word[:-1]
            if len(number) == 4: # group of years
                nonplural = ' '.join(p.number_to_words(number, group=2, wantlist=True))
                plural = p.plural(nonplural)
                words[i] = plural
            else: 
                words[i] = p.number_to_words(number) + "s"
        elif (word.endswith("th") or word.endswith("st") or word.endswith("nd") or word.endswith("rd")) and word[:-2].isdigit(): # ordinal numbers
            number = word[:-2]
            words[i] = p.ordinal(p.number_to_words(number))
        elif word[:-1].isdigit():
            number = word[:-1]
            if len(number) == 4:
                words[i] = ' '.join(p.number_to_words(number, group=2, wantlist=True)) + word[-1]
            else:
                words[i] = p.number_to_words(number) + word[-1]
        elif word.startswith("$") and is_number(word[1:]): # dollar amount
            number = word[1:]
            words[i] = p.number_to_words(number, decimal = "point") + " dollar" + ("s" if number != "1" else "")
        elif is_number(word): # number with commas and/or decimals
            words[i] = p.number_to_words(word, decimal = "point")
        else: # numbers interspersed with letters or symbols
            num = ""
            new_word = ""
            for j in range(len(word)):
                if word[j].isdigit():
                    num += word[j]
                else:
                    new_word += word[j]
                    if num != "":
                        new_word += p.number_to_words(num)
                        num = ""                
                    break
            continue

        # add punctuation back to the beginning and end of the word
        words[i] += post_punctuation
        words[i] = pre_punctuation + words[i]
    return ' '.join(words)

# in a string, remove all periods that are not at the end of a sentence
# str, Language -> str
def remove_periods(text, nlp): 
    # Process the text
    doc = nlp(text)
    
    # Go through every word in the text
    new_text = ""
    for token in doc:
        # If word is a proper noun and not the last word in the text
        if token.is_title and token.i<len(doc) - 1: 
            new_text += token.text_with_ws.replace(".", "")
        # Else keep the word as it is
        else:
            new_text += token.text_with_ws
    return new_text

# in a string, replace all abbreviations with seperate words
# str -> str
def expand_acronyms(text):
    words = text.split()

    for i in range(len(words)):
        word = words[i]

        # remove punctuation from the beginning and end of the word
        post_punctuation = ""
        while len(word) > 0 and not word[-1].isalnum():
            post_punctuation += word[-1]
            word = word[:-1]

        post_punctuation = post_punctuation[::-1]

        if len(word) == 0:
            continue

        pre_punctuation = ""
        while not word[0].isalnum():
            pre_punctuation += word[0]
            word = word[1:]

        # ignore uppercase words which also have uppercase word neighbors
        if (i > 0 and words[i-1].isupper()) or (i < len(words) - 1 and words[i+1].isupper()):
            continue
        # normal abbreviations
        elif word.isupper() and len(word) > 1:
            words[i] = ' '.join(word)
        # abbreviations that are plural
        elif word[-1] == "s" and word[:-1].isupper() and len(word) > 2:
            words[i] = ' '.join(word[:-1]) + "s"
        else:
            continue

        # add punctuation back to the beginning and end of the word
        words[i] += post_punctuation
        words[i] = pre_punctuation + words[i]
    return ' '.join(words)

# in a string, remove all parentheses and replace them with commas
# str -> str
def remove_parentheses(text):
    text = text.replace(". (", ". ")
    text = text.replace(", (", ", ")
    text = text.replace("! (", "! ")
    text = text.replace("? (", "? ")
    text = text.replace(" (", ", ")

    text = text.replace(").", ".")
    text = text.replace("),", ",")
    text = text.replace(")!", "!")
    text = text.replace(")?", "?")
    text = text.replace(") ", ", ")
    text = text.replace(")", ",")

    return text

# in a string, remove unusual punctuation and replace it
# str -> str
def reformat_punctuation(text):
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    text = text.replace("--", ", ")
    text = text.replace("-", " ")
    text = text.replace("…", "...")

    return text

# in a string, replace symbols with words
# str -> str
def reformat_symbols(text):
    text = text.replace("=", " equals ")
    text = text.replace("+", " plus ")
    text = text.replace("-", " minus ")
    text = text.replace("&", " and ") 
    text = text.replace("%", " percent ")

    return text

# in a string, replace titles with words
# str -> str
def reformat_titles(text):
    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Missus")
    text = text.replace("Ms.", "Miss")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("Prof.", "Professor")
    text = text.replace("St.", "Saint")
    text = text.replace("Mt.", "Mount")
    text = text.replace("Ft.", "Fort")

    return text

# convert a string for use with TTS
# str, Language -> str
def process_text(text, nlp):
    text = remove_parentheses(text)
    text = reformat_punctuation(text)
    text = reformat_symbols(text)
    text = reformat_titles(text)

    text = replace_numbers_with_words(text)
    text = text.replace("-", " ")
    text = remove_periods(text, nlp)
    text = expand_acronyms(text)

    text = text.replace("  ", " ")
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")

    return text

# given a string and a divider, find the largest substrings of the string
# that are on either side of the divider
# str, str -> [str]
def find_largest_substrings(text, divider):
    if divider not in text:
        return [text]
    
    substr1 = text[:len(text)/2]
    substr2 = text[len(text)/2:]

    index_right = substr1.rfind(divider)
    index_left = substr2.find(divider)

    if index_right == -1:
        substr1 = text[:index_left]
        substr2 = text[index_left:]
    elif index_left == -1:
        substr1 = text[:index_right]
        substr2 = text[index_right:]
    elif index_right > index_left:
        substr1 = text[:index_right]
        substr2 = text[index_right:]
    else:
        substr1 = text[:index_left]
        substr2 = text[index_left:]
    
    return [substr1, substr2]

dividers = [[";", ","],
            ["and", "or", "but"],
            ["because", " "]]

# given a sentence, return a list of "good" divisions of the sentence
# str -> [str]
def sentence_division(sentence):
    # if the sentence is too long, divide it into smaller sentences
    if len(sentence) > 150:
        possible_divisions = []

        for i, divider_level in enumerate(dividers):
            for divider in divider_level:
                if divider in sentence:
                    possible_divisions += (find_largest_substrings(sentence, divider), i)

        if len(possible_divisions) == 0:
            return [sentence]
        
        # sort the possible divisions by length and level
        possible_divisions.sort(key=lambda x: 10 * (len(dividers) - x[1] + 1) - abs(len(x[0][0]) - len(x[0][1])))

        # return the best division
        divisions = possible_divisions[0][0]

        # remove whitespace from the beginning and end of each division
        for i in range(len(divisions)):
            divisions[i] = divisions[i].strip()

        # run sentence_division on each division
        for i in range(len(divisions)):
            divisions[i] = sentence_division(divisions[i])

        # remove empty divisions
        divisions = list(filter(lambda x: x != "", divisions))
        
        return divisions
    else:
        return [sentence]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="epub2m4b",
        description="Convert an epub to m4b audiobook using Bark."
    )

    parser.add_argument('epub_path', type=str, help='Path to the epub file to convert.')
    parser.add_argument('-o', '--output', type=str, help='Path to the output m4b file.', default="output.m4b")
    parser.add_argument('-d', '--directory', type=str, help='Path to the directory to store the temporary files.', default="temp")
    parser.add_argument('--speaker', type=str, help='Name of the speaker.', default="v2/en_speaker_6")
    parser.add_argument('-v', '--verbose', action='store_true', help='Prints out the text as it is being processed.')

    args = parser.parse_args()

    preload_models()

    book = epub.read_epub(args.epub_path)
    
    items = []
    print("Enter the numbers of the chapter(s) you want to convert to m4b:")
    for item in book.items:
        if isinstance(item, epub.EpubHtml):
            items.append(item)
            print(str(len(items)) + ": " + item.get_name())

    chapters = input("Enter the chapter numbers separated by dashes for a range or commas: ")
    chapters = chapters.split(",")

    chapter_indices = []
    for chapter in chapters:
        if "-" in chapter:
            chapter_range = chapter.split("-")
            chapter_range = range(int(chapter_range[0]) - 1, int(chapter_range[1]))
            for i in chapter_range:
                chapter_indices.append(i)
        else:
            chapter_indices.append(int(chapter) - 1)

    current_chapter_audio = []
    nlp = spacy.load("en_core_web_sm")

    div_silence = np.zeros(int(.05 * SAMPLE_RATE))
    sentence_silence = np.zeros(int(.15 * SAMPLE_RATE))
    paragraph_silence = np.zeros(int(.5 * SAMPLE_RATE))

    class EpubHTMLParser(HTMLParser):
        def __init__(self, nlp):
            super().__init__()
            self.current_text = ""
            self.nlp = nlp
            self.ignore_next = False
        def handle_starttag(self, tag, attrs):
            if tag == "sup":
                self.ignore_next = True
        def handle_endtag(self, tag):
            if tag == "p" and not self.current_text.strip() == "":
                self.current_text = process_text(self.current_text, self.nlp)
                texts = nltk.sent_tokenize(self.current_text)
                if args.verbose:
                    print("\nParagraph: " + self.current_text + "\n")
                for text in texts:
                    for div in sentence_division(text):
                        if args.verbose:
                            print("Rendering \"" + div + "\"...")

                        semantic_tokens = generate_text_semantic(
                            div,
                            history_prompt=args.speaker,
                            min_eos_p=0.05, # TODO: tune this
                        )

                        current_chapter_audio.append(semantic_to_waveform(semantic_tokens, history_prompt=args.speaker))
                        current_chapter_audio.append(div_silence.copy())
                    current_chapter_audio.append(sentence_silence.copy())
                self.current_text = ""
                current_chapter_audio.append(paragraph_silence.copy())
            else:
                self.current_text += " "
        def handle_data(self, data):
            if not self.ignore_next:
                self.current_text += data
            else:
                self.ignore_next = False

    chapter_number = 1
    for (i, item) in enumerate(items):
        if isinstance(item, epub.EpubHtml) and i in chapter_indices:
            print(item.get_name())

            parser = EpubHTMLParser(nlp)
            parser.feed(item.get_body_content().decode("utf-8"))

            audio = np.concatenate(current_chapter_audio)
            f = os.path.join(args.directory, "chapter_" + str(chapter_number) + ".wav")
            wavfile.write(f, SAMPLE_RATE, audio)

            # use ffmpeg to convert to mp3
            subprocess.run(["ffmpeg", "-i", f, f.replace(".wav", ".mp3")])

            current_chapter_audio = []
            chapter_number += 1

    # use ffmpeg to combine the mp3s into one m4b
    concat_string = ""
    for i in range(1, chapter_number):
        concat_string += args.directory + "chapter_" + str(i) + ".mp3|"
    concat_string = concat_string[:-1]

    subprocess.run(["ffmpeg", "-i", "concat:" + concat_string, "-f", "mp4", args.output])