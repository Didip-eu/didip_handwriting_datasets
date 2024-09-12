import pytest
import sys
import torch
from torch import Tensor
from pathlib import Path

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

from handwriting_datasets import alphabet

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')

@pytest.fixture(scope="session")
def alphabet_one_to_one_tsv(data_path):
    return data_path.joinpath('alphabet_one_to_one_repr.tsv')


def test_alphabet_from_string():
    """
    Unique characters, Unicode legit, sorted, nullspace, other space chars ignored
    """
    # null char
    alpha = alphabet.Alphabet.from_string('ßafdbce→')
    assert alpha['∅']==0
    # unique symbols, sorted
    assert alphabet.Alphabet.from_string('ßaafdbce →e') == { '∅': 0, ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'ß': 8, '→': 9}
    # space chars ignored
    assert alphabet.Alphabet.from_string('ßaf \u2009db\n\tce\t→') == { '∅': 0, ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'ß': 8, '→': 9}


def test_alphabet_from_dict():
    """
    Unique characters, Unicode legit, sorted, nullspace, other space chars ignored
    """
    # null char
    alpha = alphabet.Alphabet.from_dict( { ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'ß': 8, '→': 9})
    # unique symbols, sorted
    assert alpha == { '∅': 0, ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'ß': 8, '→': 9}


def test_alphabet_from_tsv( alphabet_one_to_one_tsv ):
    """
    Unique characters, Unicode legit, sorted, nullspace, other space chars ignored
    """
    # null char
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_one_to_one_tsv) )
    # unique symbols, sorted
    assert alpha == {'∅': 0, ' ': 1, ',': 2, 'A': 3, 'J': 10, 'R': 15, 'S': 16, 'V': 17, 'b': 20, 'c': 21, 'd': 22, 'o': 32, 'p': 33, 'r': 34, 'w': 39, 'y': 40, 'z': 41, '¬': 42, 'ü': 43}


def test_alphabet_init_from_str():
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→')
    assert alpha._utf_2_code == {' ':1, 'a':2, 'b':3, 'c':4, 'd':5, 'e':6, 'f':7, 'ß':8, '→':9, '∅':0}
    assert alpha._code_2_utf == {1:' ', 2:'a', 3:'b', 4:'c', 5:'d', 6:'e', 7:'f', 8:'ß', 9:'→', 0:'∅'}


def test_alphabet_init_from_tsv( alphabet_one_to_one_tsv ):
    alpha = alphabet.Alphabet( str(alphabet_one_to_one_tsv) )
    assert alpha._utf_2_code == {'∅': 0, ' ': 1, ',': 2, 'A': 3, 'J': 10, 'R': 15, 'S': 16,
                                 'V': 17, 'b': 20, 'c': 21, 'd': 22, 'o': 32, 'p': 33, 'r': 34,
                                 'w': 39, 'y': 40, 'z': 41, '¬': 42, 'ü': 43}
    assert alpha._code_2_utf == {0:'∅', 1:' ', 2:',', 3:'A', 10:'J', 15:'R', 16:'S',
                                 17:'V', 20:'b', 21:'c', 22:'d', 32:'o', 33:'p', 34:'r',
                                 39:'w', 40:'y', 41:'z', 42:'¬', 43:'ü'}
    
def test_alphabet_init_from_dict():
    alpha = alphabet.Alphabet( { ' ':1, 'a':2, 'b':3, 'c':4, 'd':5, 'e':6, 'f':7, 'ß':8, '→':9} )
    assert alpha._code_2_utf == {1:' ', 2:'a', 3:'b', 4:'c', 5:'d', 6:'e', 7:'f', 8:'ß', 9:'→', 0:'∅'}
    assert alpha._utf_2_code == {' ':1, 'a':2, 'b':3, 'c':4, 'd':5, 'e':6, 'f':7, 'ß':8, '→':9, '∅':0}

def test_alphabet_len():

    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert len( alpha ) == 10


def test_alphabet_contains_symbol():
    """ 'in' operator """
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert 'a' in alpha
    assert 'z' not in alpha

def test_alphabet_contains_code():

    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert 1 in alpha
    assert 43 not in alpha

def test_alphabet_get_symbol():
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert alpha.get_symbol( 8 ) == 'ß'

def test_alphabet_get_code():
    alpha = alphabet.Alphabet('ßafdb\n\tce\t→') 
    assert alpha.get_code( 'ß' ) == 7
    assert alpha.get_code( 'z' ) == 0

def test_alphabet_getitem():
    """ Subscript access """
    alpha = alphabet.Alphabet('ßa fdb\n\tce\t→') 
    print(alpha)
    assert alpha['ß'] == 8
    assert alpha[8] == 'ß'

def test_alphabet_eq():
    """ Testing for equality """
    alpha1= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    alpha2= alphabet.Alphabet('ßa fdb\n\tce\t→')
    alpha3= alphabet.Alphabet('ßa db\n\tce\t→')
    assert alpha1 == alpha2
    assert alpha1 != alpha3


def test_encode_clean_sample():
    alpha= alphabet.Alphabet('ßa fdbce→') 
    encoded = alpha.encode('abc ß def ')
    assert encoded.equal( torch.tensor([2, 3, 4, 1, 8, 1, 5, 6, 7, 1], dtype=torch.int64))

def test_encode_missing_symbols():
    """Unknown symbols generate null char (and a warning)."""
    alpha= alphabet.Alphabet('ßa fdbce→') 
    with pytest.warns(UserWarning):
        encoded = alpha.encode('abc z def ')
        assert encoded.equal( torch.tensor([2, 3, 4, 1, 0, 1, 5, 6, 7, 1], dtype=torch.int64))

def test_encode_illegal_symbols():
    """Illegal symbols raise an exception."""
    alpha= alphabet.Alphabet('ßa fdbce→') 
    with pytest.raises(ValueError):
        encoded = alpha.encode('abc\n z def ')
    with pytest.raises(ValueError):
        encoded = alpha.encode('abc\t z def ')

def test_encode_one_hot():
    alpha= alphabet.Alphabet('ßa fdbce→') 
    assert alpha.encode_one_hot('abc ß def ').equal( torch.tensor(
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool ))

def test_decode():
    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    # full length (default)
    assert alpha.decode( torch.tensor([2, 3, 4, 1, 9, 1, 5, 6, 7, 1], dtype=torch.int64 )) == 'abc → def '
    # explicit length
    assert alpha.decode( torch.tensor([2, 3, 4, 1, 9, 1, 5, 6, 7, 1], dtype=torch.int64 ), 5) == 'abc →'


def test_decode_unknown_code():
    """ TODO """
    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    # full length (default)
    assert alpha.decode( torch.tensor([2, 3, 4, 1, 9, 1, 5, 6, 7, 1], dtype=torch.int64 )) == 'abc → def '
    # explicit length
    assert alpha.decode( torch.tensor([2, 3, 4, 1, 9, 1, 5, 6, 7, 1], dtype=torch.int64 ), 5) == 'abc →'




def test_encode_batch_1():
    """ Batch with clean strings """
    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    batch_str = [ 'abc def ', 'ßecbca ' ]
    encoded = alpha.encode_batch( batch_str )

    assert encoded[0].equal( 
            torch.tensor( [[2, 3, 4, 1, 5, 6, 7, 1],
                           [8, 6, 4, 3, 4, 2, 1, 0]], dtype=torch.int64))
    assert encoded[1].equal( 
            torch.tensor([8,7], dtype=torch.int64 ))


def test_decode_batch():

    alpha= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    print(alpha)
    samples, lengths = (torch.tensor( [[2, 3, 4, 1, 5, 6, 7, 1],
                            [8, 6, 4, 3, 4, 2, 1, 0]], dtype=torch.int64),
             torch.tensor( [8, 7]))
    assert alpha.decode_batch( samples, lengths ) == ["abc def ", "ßecbca "]
    assert alpha.decode_batch( samples, None ) == ["abc def ", "ßecbca ∅"]



def test_dummy( data_path):
    """
    A dummy test, as a sanity check for the test framework.
    """
    print(data_path)
    assert alphabet.dummy() == True

