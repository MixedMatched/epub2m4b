import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ebooklib import epub
import inflect
import spacy
import nltk
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import SAMPLE_RATE

import argparse
from html.parser import HTMLParser

months = ["January", "February", "March", 
            "April", "May", "June", 
            "July", "August", "September",
            "October", "November", "December"]

def is_number(word):
    for char in word:
        if not (char.isdigit() or char == "." or char == ","):
            return False
    return True

def replace_numbers_with_words(text):
    p = inflect.engine()
    words = text.split()
    for i in range(len(words)):
        word = words[i]

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
        elif (word.endswith("th") or word.endswith("st") or word.endswith("nd") or word.endswith("rd")) and word[:-2].isdigit(): # ordinal number
            number = word[:-2]
            words[i] = p.ordinal(p.number_to_words(number))
        elif word[:-1].isdigit():
            number = word[:-1]
            if len(number) == 4:
                words[i] = ' '.join(p.number_to_words(number, group=2, wantlist=True)) + word[-1]
            else:
                words[i] = p.number_to_words(number) + word[-1]
        elif word.startswith("$") and is_number(word[1:]): 
            number = word[1:]
            words[i] = p.number_to_words(number, decimal = "point") + " dollar" + ("s" if number != "1" else "")
        elif is_number(word):
            words[i] = p.number_to_words(word, decimal = "point")
        else:
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
        words[i] += post_punctuation
        words[i] = pre_punctuation + words[i]
    return ' '.join(words)

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

def expand_acronyms(text):
    words = text.split()

    for i in range(len(words)):
        word = words[i]

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

        if (i > 0 and words[i-1].isupper()) or (i < len(words) - 1 and words[i+1].isupper()):
            continue
        elif word.isupper() and len(word) > 1:
            words[i] = ' '.join(word)
        elif word[-1] == "s" and word[:-1].isupper() and len(word) > 2:
            words[i] = ' '.join(word[:-1]) + "s"
        else:
            continue

        words[i] += post_punctuation
        words[i] = pre_punctuation + words[i]
    return ' '.join(words)

def process_text(text, nlp):
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    text = text.replace("--", ", ")
    text = text.replace("-", " ")
    text = text.replace("…", "...")
    text = text.replace("=", " equals ")
    text = text.replace("+", " plus ")
    text = text.replace("-", " minus ")

    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Missus")
    text = text.replace("Ms.", "Miss")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("Prof.", "Professor")
    text = text.replace("St.", "Saint")
    text = text.replace("Mt.", "Mount")
    text = text.replace("Ft.", "Fort")

    text = replace_numbers_with_words(text)
    text = text.replace("-", " ")
    text = remove_periods(text, nlp)
    text = expand_acronyms(text)

    return text

# given a sentence, return a list of "good" divisions of the sentence
def sentence_division(sentence):
    if len(sentence) < 150:
        divisions = []

        if ";" in sentence:
            for division in sentence.split(";"):
                divisions += sentence_division(division)
        elif "," in sentence:
            for division in sentence.split(","):
                divisions += sentence_division(division)
        else:
            division_counter = 0
            divisions.append("")
            for word in sentence.split(" "):
                divisions[division_counter] += word + " "
                if len(divisions[division_counter]) > 125:
                    division_counter += 1
                    divisions.append("")

        division = filter(lambda x: x != "", divisions)
        
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
            print(len(items), ": ", item.get_name())

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
        def handle_endtag(self, tag):
            if tag == "p" and not self.current_text.strip() == "":
                texts = nltk.sent_tokenize(self.current_text)
                if args.verbose:
                    print("Paragraph: " + self.current_text)
                for text in texts:
                    for div in sentence_division(text):
                        if args.verbose:
                            print("Rendering " + div + "...")

                        semantic_tokens = generate_text_semantic(
                            div,
                            history_prompt=args.speaker,
                            min_eos_p=0.05,
                        )

                        current_chapter_audio.append(semantic_to_waveform(semantic_tokens, history_prompt=args.speaker))
                        current_chapter_audio.append(div_silence.copy())
                    current_chapter_audio.append(sentence_silence.copy())
                self.current_text = ""
                current_chapter_audio.append(paragraph_silence.copy())
            else:
                self.current_text += " "
        def handle_data(self, data):
            data = process_text(data, self.nlp)
            self.current_text += data

    chapter_number = 1
    for (i, item) in enumerate(items):
        if isinstance(item, epub.EpubHtml) and i in chapter_indices:
            print(item.get_name())

            parser = EpubHTMLParser(nlp)
            parser.feed(item.get_body_content().decode("utf-8"))

            audio = np.concatenate(current_chapter_audio)
            f = "chapter_" + str(chapter_number) + ".wav"
            wavfile.write(f, SAMPLE_RATE, audio)

            current_chapter_audio = []
            chapter_number += 1
