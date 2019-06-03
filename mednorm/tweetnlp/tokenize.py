from six import string_types

from mednorm.tweetnlp.ark.twokenize import tokenizeRawTweetText


def word_tokenize(text, with_spans=False):
    tokens = tokenizeRawTweetText(text)
    if with_spans:
        return extract_tokens_spans(text, tokens)
    return tokens


def extract_tokens_spans(text, tokens):
    proc_tokens = list(tokens)
    ind = 0
    for i in range(len(proc_tokens)):
        t = proc_tokens[i]
        if isinstance(t, string_types):
            t = [t]
        elif isinstance(t, tuple):
            t = list(t)
        try:
            ind = text.find(t[0], ind)
        except ValueError:
            raise ValueError("Token %s is not found in %s" % (t[0], text))
        t.append((ind, ind + len(t[0])))
        ind += len(t[0])
        proc_tokens[i] = t
    return proc_tokens
