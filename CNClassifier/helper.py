import string,re


def puncRemove(note):
    # Punctuations remove
    punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))
    text= punctuation_regex.sub("", note)
    return text

def note_cleanser(note):
    return puncRemove(note.lower())

