import pytest
import sys
import torch
import numpy as np
from torch import Tensor
from pathlib import Path
import random

# Append app's root directory to the Python search path
sys.path.append( str( Path(__file__).parents[1] ) )

from didip_handwriting_datasets import alphabet

@pytest.fixture(scope="session")
def data_path():
    return Path( __file__ ).parent.joinpath('data')

@pytest.fixture(scope="session")
def alphabet_one_to_one_tsv(data_path):
    return data_path.joinpath('alphabet_one_to_one_repr_without_nullchar.tsv')

@pytest.fixture(scope="session")
def alphabet_one_to_one_tsv_nullchar(data_path):
    return data_path.joinpath('alphabet_one_to_one_repr_with_nullchar.tsv')

@pytest.fixture(scope="session")
def alphabet_many_to_one_tsv(data_path):
    return data_path.joinpath('alphabet_many_to_one_repr.tsv')

@pytest.fixture(scope="session")
def alphabet_many_to_one_prototype_tsv( data_path ):
    return data_path.joinpath('alphabet_many_to_one_prototype.tsv')

@pytest.fixture(scope="session")
def gt_transcription_samples( data_path ):
    return [ str(data_path.joinpath(t)) for t in ('transcription_sample_1.gt.txt', 'transcription_sample_2.gt.txt') ]

def test_alphabet_dict_from_string():
    """
    Raw dictionary reflects the given string: no less, no more; no virtual chars (null, ⇥, ...)
    """
    # unique symbols, sorted
    assert alphabet.Alphabet.from_string('ßaafdbce →e') == {' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'ß': 8, '→': 9, }
    # space chars ignored
    assert alphabet.Alphabet.from_string('ßaf \u2009db\n\tce\t→') == {' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'ß': 8, '→': 9}


def test_alphabet_dict_from_dict():
    """
    Raw dictionary reflects the given string: no less, no more; no virtual chars (null, ⇥, ...)
    """
    # null char
    alpha = alphabet.Alphabet.from_dict( { ' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'ß': 8, '→': 9})
    # unique symbols, sorted
    assert alpha == {' ': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'ß': 8, '→': 9}


def test_alphabet_dict_from_tsv_with_null_char( alphabet_one_to_one_tsv_nullchar ):
    """
    Raw dict contains everything than what is in the TSV
    """
    # null char
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_one_to_one_tsv_nullchar) )
    # unique symbols, sorted
    assert alpha == {'ϵ': 0, ' ': 1, ',': 2, 'A': 3, 'J': 10, 'R': 15, 'S': 16, 'V': 17,
                     'b': 20, 'c': 21, 'd': 22, 'o': 32, 'p': 33, 'r': 34, 'w': 39, 
                     'y': 40, 'z': 41, '¬': 42, 'ü': 43}

def test_alphabet_dict_from_tsv_without_null_char( alphabet_one_to_one_tsv ):
    """
    Raw dict contains nothing more than what is in the TSV
    """
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_one_to_one_tsv) )
    # unique symbols, sorted
    assert alpha == {' ': 1, ',': 2, 'A': 3, 'J': 10, 'R': 15, 'S': 16, 'V': 17,
                     'b': 20, 'c': 21, 'd': 22, 'o': 32, 'p': 33, 'r': 34, 'w': 39, 
                     'y': 40, 'z': 41, '¬': 42, 'ü': 43}

def test_alphabet_from_list_one_to_one():
    input_list = ['A', 'a', 'J', 'b', 'ö', 'o', 'O', 'ü', 'U', 'w', 'y', 'z', 'd', 'D']
    alpha = alphabet.Alphabet.from_list( input_list )
    assert alpha == {'A': 1, 'D': 2, 'J': 3, 'O': 4, 'U': 5, 'a': 6, 'b': 7, 'd': 8, 'o': 9, 'w': 10, 'y': 11, 'z': 12, 'ö': 13, 'ü': 14}

def test_alphabet_from_list_compound_symbols_one_to_one():
    input_list = ['A', 'ae', 'J', 'ü', 'eu', 'w', 'y', 'z', '...', 'D']
    alpha = alphabet.Alphabet.from_list( input_list )
    assert alpha == {'...': 1, 'A': 2, 'D': 3, 'J': 4, 'ae': 5, 'eu': 6, 'w': 7, 'y': 8, 'z': 9, 'ü': 10 }

def test_alphabet_from_list_many_to_one():
    input_list = [['A', 'a'], 'J', 'b', ['ö', 'o', 'O'], 'ü', 'U', 'w', 'y', 'z', ['d', 'D']]
    alpha = alphabet.Alphabet.from_list( input_list )
    assert alpha == {'A': 1, 'D': 2, 'J': 3, 'O': 4, 'U': 5, 'a': 1, 'b': 6, 'd': 2, 'o': 4, 'w': 7, 'y': 8, 'z': 9, 'ö': 4, 'ü': 10}
                    
def test_alphabet_from_list_realistic():
    """ Passing a list with virtual symbols (EoS, SoS) yields a correct mapping 
    """
    input_list = [ ' ', ',', '-', '.', '1', '2', '4', '5', '6', ':', ';', ['A', 'a', 'ä'], ['B', 'b'], ['C', 'c'], [    'D', 'd'], ['E', 'e', 'é'], '⇥', ['F', 'f'], ['G', 'g'], ['H', 'h'], ['I', 'i'], ['J', 'j'], ['K', 'k'], ['L', 'l'], ['M', 'm'], ['N', 'n'], ['O', 'o', 'Ö', 'ö'], ['P', 'p'], ['Q', 'q'], ['R', 'r', 'ř'], ['S', 's'], '↦', ['T', 't'], ['U', 'u', 'ü'], ['V', 'v'], ['W', 'w'], ['X', 'x'], ['Y', 'y', 'ÿ'], ['Z', 'z', 'Ž'], '¬','…' ]
    alpha = alphabet.Alphabet.from_list( input_list )
    assert alpha == {' ': 1, ',': 2, '-': 3, '.': 4, '1': 5, '2': 6, '4': 7, '5': 8, '6': 9, ':': 10, ';': 11, 'A': 12, 'B': 13, 'C': 14, 'D': 15, 'E': 16, 'F': 17, 'G': 18, 'H': 19, 'I': 20, 'J': 21, 'K': 22, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 33, 'W': 34, 'X': 35, 'Y': 36, 'Z': 37, 'a': 12, 'b': 13, 'c': 14, 'd': 15, 'e': 16, 'f': 17, 'g': 18, 'h': 19, 'i': 20, 'j': 21, 'k': 22, 'l': 23, 'm': 24, 'n': 25, 'o': 26, 'p': 27, 'q': 28, 'r': 29, 's': 30, 't': 31, 'u': 32, 'v': 33, 'w': 34, 'x': 35, 'y': 36, 'z': 37, '¬': 38, 'Ö': 26, 'ä': 12, 'é': 16, 'ö': 26, 'ü': 32, 'ÿ': 36, 'ř': 29, 'Ž': 37, '…': 39}


def test_alphabet_many_to_one_from_tsv( alphabet_many_to_one_tsv ):
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_many_to_one_tsv) )
    # unique symbols, sorted
    assert alpha == {'ϵ': 0, 'A': 1, 'D': 10, 'J': 2, 'O': 4, 'U': 6, 'a': 1, 'b': 3, 
                     'd': 10, 'o': 4, 'w': 7, 'y': 8, 'z': 9, 'ö': 4, 'ü': 5}

def test_alphabet_many_to_one_prototype_tsv( alphabet_many_to_one_prototype_tsv ):
    alpha = alphabet.Alphabet.from_tsv( str(alphabet_many_to_one_prototype_tsv), prototype=True)
    assert alpha == {'A': 1, 'D': 2, 'J': 3, 'O': 4, 'U': 5, 'ae': 1, 'b': 6, 'd': 2, 'o': 4,
                    'w': 7, 'y': 8, 'z': 9, 'ö': 4, 'ü': 10}


def test_alphabet_many_to_one_init( alphabet_many_to_one_tsv ):
    alpha = alphabet.Alphabet( str(alphabet_many_to_one_tsv) )
    # unique symbols, sorted
    assert alpha._utf_2_code == {'ϵ': 0, 'A': 1, 'D': 10, 'J': 2, 'O': 4, 'U': 6, 'a': 1, 'b': 3, 
            'd': 10, 'o': 4, 'w': 7, 'y': 8, 'z': 9, 'ö': 4, 'ü': 5, '↦': 11, '⇥': 12}
    assert alpha._code_2_utf == {0: 'ϵ', 1: 'a', 10: 'd', 2: 'j', 4: 'o', 6: 'u', 3: 'b', 7: 'w', 8: 'y', 9: 'z', 5: 'ü', 11: '↦', 12: '⇥'}

def test_alphabet_many_to_one_deterministic_tsv_init(data_path):
    """ Given a code, a many-to-one alphabet from tsv consistently returns the same symbol,
        no matter the order of the items in the input file.
    """
    # initialization from the same input (TSV here) give consistent results
    symbols = set()
    for i in range(10):
        symbols.add( alphabet.Alphabet( str(data_path.joinpath('lol_many_to_one_shuffled_{}'.format(i))) ).get_symbol(1))
    assert len(symbols) == 1


def test_alphabet_many_to_one_deterministic_dict_init():
    """
    Initialization from dictionaries in different orders (but same mapping) gives consistent results
    """
    key_values = [ ('A',1), ('D',10), ('J',2), ('O',4), ('U',6), ('a',1), ('b',3), ('d',10), ('o',4), ('w',7), ('y',8), ('z',9), ('ö',4), ('ü',5) ]
    symbols = set()
    for i in range(10):
        random.shuffle( key_values )
        symbols.add( alphabet.Alphabet( { k:v for (k,v) in key_values } ).get_symbol(1) )
    assert len(symbols) == 1


def test_alphabet_many_to_one_deterministic_list_init():
    """ 
    Initialization from lists in different orders (but same k,v) give consistent results
    """
    list_of_lists = [['A', 'a'], 'J', 'b', ['ö', 'o', 'O'], 'ü', 'U', 'w', 'y', 'z', ['d', 'D']]
    symbols = set()
    for i in range(10):
        random.shuffle( list_of_lists )
        symbols.add( alphabet.Alphabet( list_of_lists ).get_symbol(1) )
    assert len(symbols) == 1

def test_alphabet_many_to_one_compound_symbols_deterministic_list_init():
    """
    Initialization from lists in different orders (but same k,v) give consistent results
    (testing with compound symbols)
    """
    list_of_lists = [['A', 'ae'], 'b', ['ü', 'ue', 'u', 'U'], 'c']
    symbols = set()
    for i in range(10):
        random.shuffle( list_of_lists )
        symbols.add( alphabet.Alphabet( list_of_lists ).get_symbol(1) )
    assert len(symbols) == 1

def test_alphabet_many_to_one_deterministic_different_input_methods( data_path ):
    """
    Different initialization methods yield deterministic code retrieval behaviour.
    """
    symbols = set()
    list_of_lists = [['A', 'a'], 'J', 'b', ['ö', 'o', 'O'], 'ü', 'U', 'w', 'y', 'z', ['d', 'D']]
    for i in range(5):
        random.shuffle( list_of_lists )
        symbols.add( alphabet.Alphabet( list_of_lists ).get_symbol(1) )

    key_values = [ ('A',1), ('D',10), ('J',2), ('O',4), ('U',6), ('a',1), ('b',3), ('d',10), ('o',4), ('w',7), ('y',8), ('z',9), ('ö',4), ('ü',5) ]
    for i in range(10):
        random.shuffle( key_values )
        symbols.add( alphabet.Alphabet( { k:v for (k,v) in key_values } ).get_symbol(1) )
    assert len(symbols) == 1

def test_alphabet_init_from_str():
    alpha = alphabet.Alphabet('ßaf db\n\tce\t→')
    assert alpha._utf_2_code == {' ':1, 'a':2, 'b':3, 'c':4, 'd':5, 'e':6, 'f':7, 'ß':8, '→':9, 'ϵ':0,
                                '↦': 10, '⇥': 11}
    assert alpha._code_2_utf == {1:' ', 2:'a', 3:'b', 4:'c', 5:'d', 6:'e', 7:'f', 8:'ß', 9:'→', 0:'ϵ',
                                 10:'↦', 11:'⇥'}


def test_alphabet_init_from_tsv( alphabet_one_to_one_tsv ):
    alpha = alphabet.Alphabet( str(alphabet_one_to_one_tsv) )
    assert alpha._utf_2_code == {'ϵ': 0, ' ': 1, ',': 2, 'A': 3, 'J': 10, 'R': 15, 'S': 16,
                                 'V': 17, 'b': 20, 'c': 21, 'd': 22, 'o': 32, 'p': 33, 'r': 34,
                                 'w': 39, 'y': 40, 'z': 41, '¬': 42, 'ü': 43, '↦': 44, '⇥': 45}
    assert alpha._code_2_utf == {0:'ϵ', 1:' ', 2:',', 3:'A', 10:'J', 15:'R', 16:'S',
                                 17:'V', 20:'b', 21:'c', 22:'d', 32:'o', 33:'p', 34:'r',
                                 39:'w', 40:'y', 41:'z', 42:'¬', 43:'ü', 44:'↦', 45:'⇥'}
    
def test_alphabet_init_from_dict():
    alpha = alphabet.Alphabet( { ' ':1, 'a':2, 'b':3, 'c':4, 'd':5, 'e':6, 'f':7, 'ß':8, '→':9} )
    assert alpha._utf_2_code == {' ':1, 'a':2, 'b':3, 'c':4, 'd':5, 'e':6, 'f':7, 'ß':8, '→':9,
                                 'ϵ':0, '↦': 10, '⇥': 11}
    assert alpha._code_2_utf == {1:' ', 2:'a', 3:'b', 4:'c', 5:'d', 6:'e', 7:'f', 8:'ß', 9:'→',
                                 0:'ϵ', 10:'↦', 11:'⇥'}

def test_alphabet_to_list():
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'o', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']

    #def deep_sorted(l):
    #    return sorted([ sorted(item) if len(item)>1 else item[0] for item in l ],
    #            key=lambda x: x[0])
    assert alphabet.Alphabet( list_of_lists ).to_list() == list_of_lists



def test_alphabet_to_list_minus_symbols():
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'o', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']

    assert alphabet.Alphabet( list_of_lists ).to_list(exclude=['o','w']) == [['A', 'a'], ['D', 'd'], 'J', ['O', 'ö'], 'U', 'b', 'y', 'z', 'ü']


def test_alphabet_remove_symbol_1():
    alpha = alphabet.Alphabet( [['A', 'a'], ['D', 'd'], 'J', ['O', 'o', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü'] )
    
    alpha.remove_symbols(['o', 'w'])
    assert alpha._utf_2_code == {'A': 1, 'D': 2, 'J': 3, 'O': 4, 'U': 5, 'a': 1, 'b': 6, 'd': 2, 'y': 7, 'z': 8, 'ö': 4, 'ü': 9, 'ϵ': 0, '↦': 10, '⇥': 11}



def test_alphabet_add_symbols_no_merge():
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']
    alpha = alphabet.Alphabet( list_of_lists )
    alpha.add_symbols(['o', ['ÿ', 'ŷ']])
    assert alpha._utf_2_code == {'A': 1, 'D': 2, 'J': 3, 'O': 4, 'U': 5, 'a': 1, 'b': 6, 'd': 2, 'o': 7, 'w': 8, 'y': 9, 'z': 10, 'ö': 4, 'ü': 11, 'ÿ': 12, 'ŷ': 12, 'ϵ': 0, '↦': 13, '⇥': 14}


def test_alphabet_add_symbols_merging_subgroups():
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']
    alpha = alphabet.Alphabet( list_of_lists )
    alpha.add_symbols(['o', ['ÿ', 'ŷ', 'y']])
    assert alpha._utf_2_code == {'A': 1, 'D': 2, 'J': 3, 'O': 4, 'U': 5, 'a': 1, 'b': 6, 'd': 2, 'o': 7, 'w': 8, 'y': 9, 'z': 10, 'ö': 4, 'ü': 11, 'ÿ': 9, 'ŷ': 9, 'ϵ': 0, '↦': 12, '⇥': 13}

def test_alphabet_add_symbols_merging_atoms():
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']
    alpha = alphabet.Alphabet( list_of_lists )
    alpha.add_symbols([['B','b'], ['ÿ', 'ŷ']])
    assert alpha._utf_2_code == {'A': 1, 'B': 2, 'D': 3, 'J': 4, 'O': 5, 'U': 6, 'a': 1, 'b': 2, 'd': 3, 'w': 7, 'y': 8, 'z': 9, 'ö': 5, 'ü': 10, 'ÿ': 11, 'ŷ': 11, 'ϵ': 0, '↦': 12, '⇥': 13}


def test_alphabet_add_symbols_faulty_hooks():
    """
    Subgroup contains symbols that already map to different codes in the alphabet.
    """
    list_of_lists = [['A', 'a'], ['D', 'd'], 'J', ['O', 'ö'], 'U', 'b', 'w', 'y', 'z', 'ü']
    alpha = alphabet.Alphabet( list_of_lists )
    with pytest.raises(ValueError) as error:
        alpha.add_symbols([['B','b','w'], ['ÿ', 'ŷ']])




def test_alphabet_len():

    alpha = alphabet.Alphabet('ßaf db\n\tce\t→') 
    assert len( alpha ) == 12


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
    assert alpha['ß'] == 8
    assert alpha[8] == 'ß'

def test_alphabet_eq():
    """ Testing for equality """
    alpha1= alphabet.Alphabet('ßa fdb\n\tce\t→') 
    alpha2= alphabet.Alphabet('ßa fdb\n\tce\t→')
    alpha3= alphabet.Alphabet('ßa db\n\tce\t→')
    assert alpha1 == alpha2
    assert alpha1 != alpha3

def test_alphabet_prototype_ascii( gt_transcription_samples ):
    list_of_list,_ = alphabet.Alphabet.prototype( gt_transcription_samples, many_to_one=True )
    assert list_of_list == [' ', ',', '.', ['B', 'b'], ['E', 'e'], ['H', 'h'], ['K', 'k'], 'P', ['U', 'u'],
                            ['W', 'w'], 'a', 'c', 'd', 'f', 'g', 'i', 'l', 'm', 'n', 'o', 'r', 's', 't', 'v', 'y', 'z']


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
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool ))

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
    samples, lengths = (torch.tensor( [[2, 3, 4, 1, 5, 6, 7, 1],
                            [8, 6, 4, 3, 4, 2, 1, 0]], dtype=torch.int64),
             torch.tensor( [8, 7]))
    assert alpha.decode_batch( samples, lengths ) == ["abc def ", "ßecbca "]
    assert alpha.decode_batch( samples, None ) == ["abc def ", "ßecbca ϵ"]


def test_decode_ctc():
    alpha = alphabet.Alphabet([' ', ',', '-', '.', '1', '2', '4', '5', '6', ':', ';', ['A', 'a', 'ä'],
                              ['B', 'b'], ['C', 'c'], ['D', 'd'], ['E', 'e', 'é'], ['F', 'f'], ['G', 'g'],
                              ['H', 'h'], ['I', 'i'], ['J', 'j'], ['K', 'k'], ['L', 'l'], ['M', 'm'],
                              ['N', 'n'], ['O', 'o', 'Ö', 'ö'], ['P', 'p'], ['Q', 'q'], ['R', 'r', 'ř'],
                              ['S', 's'], ['T', 't'], ['U', 'u', 'ü'], ['V', 'v'], ['W', 'w'], ['X', 'x'],
                              ['Y', 'y', 'ÿ'], ['Z', 'z', 'Ž'], '¬', '…'])

    decoded = alpha.decode_ctc( np.array([19, 19, 19, 0, 0, 16, 16, 0, 23, 23, 23, 23, 0, 0,
                                23, 23, 0, 26, 26, 2, 0, 1, 34, 34, 26, 29, 29, 29, 
                                23, 15]) )
    assert decoded == 'hello, world'


def atest_dummy( data_path):
    """
    A dummy test, as a sanity check for the test framework.
    """
    print(data_path)
    assert alphabet.dummy() == True

