from __future__ import annotations # to allow for type hints to reference the enclosing class

from typing import Union,Tuple,List,Dict  #,Self (>= 3.11)
import torch
from torch import Tensor
import numpy as np
import re
from pathlib import Path
import itertools
import warnings
from collections import Counter




class Alphabet:
    """
    Creating and handling alphabets.

    + one-to-one or many-to-one alphabet, with deterministic mapping either way;
    + prototyping from reasonable subsets of characters to be grouped
    + a choice of input/output sources: TSV, nested lists, mappings.

    Alphabet Philosophy or To Clarify My Ideas about encoding/decoding

    Problem: fitting data actual charset to a model

    Distinguish between:

    1. symbols we want to ignore/discard, because they're junk or 
       irrelevant for any HTR purpose - they typically do not have
       any counterpart in the input image.

    2. symbols that may have a counterpart in the original image,
       but we want to ignore now because they don't matter for our
       particular experiment - by they may do for the next one.
       It is the HTR analog of '> /dev/null' !

    3. symbols we don't want to interpret but need to be there in some way
       in the output (if just for being able to say: there is something
       here that needs interpreting. Eg. abbreviations)
    
    For (1), they can be filtered out of the sample at any stage
    between the page/xlm collation phase and the last, encoding stage,
    but it makes sense to do it as early as possible, to avoid any
    overhead later, while not having to think too much about it (i.e.
    avoid having to use a zoo of pre-processing scripts that live
    outside this module): the line extraction method seems a good place
    to do that, and no alphabet is normally needed.

    For (2) and (3), we want more flexibility: what is to be ignored or
    not interpreted now may change tomorrow. The costly line extraction
    routine, that is meant to be run during the early dataset setup and
    generally once, is not the best place to do it: rather, the 
    getitem transforms are appropriate. But how?

    * to ignore a symbol, just do not include it into the alphabet: it 
      will map to null at the encoding stage, while still occupy one slot
      in the sequence, which is what we want;
    
    * a symbol that is not to be interpreted needs to be in the alphabet,
      even though it is just a normal symbol, that maps to a conventional code
      for 'unknown'. Eg. a character class '?' may contain all tricky baseline
      abbreviations -> it must be easy to merge different classes into 
      a single 'unknown' set.

    In practice, any experiment needs to define its alphabet, as follows:

    1. Define a reasonable, all-purpose default alphabet in the ChartersDataset
       class, that leaves most options open, while still excluding the most
       unlikely ones. Eg.  an alphabet without the Hebrew characters, diacritic marks, etc.
       It should be a class property, so that any addition to the character classes
       is duly reflected in the default.

    2. Construct a specific alphabet out of it, by substraction or merging.
       A that point, the class 'unknown' is just like another class: all 
       subsets of characters that make it should be merged accordingly first.
       Only then does the finalization method define a stand-in symbol
       for this class, through a quick lookup at any character that is member
       of the set (see `unknown_class_representant` option in the initialization
       function).
    

    """
    null_symbol = '\u03f5'
    null_value = 0
    start_of_seq_symbol = '\u21A6' # '↦' i.e. '|->'
    end_of_seq_symbol = '\u21E5' # '⇥' i.e. '->|'
    unknown_symbol_utf = '?' 

    def __init__( self, alpha_repr: Union[str,dict,list]='', tokenizer=None, unknown_class_representant:str='ƺ') -> None:
        """ Initialize a new Alphabet object. The special characters are
        added automatically.

        From a dictionary:
        .. code-block::

            >>> alphabet.Alphabet({'a':1, 'A': 1, 'b': 2, 'c':3})
            {'A': 1, 'a': 1, 'b': 2, 'c': 3, 'ϵ': 0, '↦': 4, '⇥': 5}

        From a TSV path:
        .. code-block::

            >>> alphabet.Alphabet('alpha.tsv')
            {'A': 1, 'a': 1, 'b': 2, 'c': 3, 'ϵ': 0, '↦': 4, '⇥': 5}

        From a nested list:
        .. code-block::

            >>> alphabet.Alphabet([['a','A'],'b','c'])
            {'A': 1, 'a': 1, 'b': 2, 'c': 3, 'ϵ': 0, '↦': 4, '⇥': 5}

        From a string of characters (one-to-one):
        .. code-block::

            >>> alphabet.Alphabet('aAbc ')
            {' ': 1, 'A': 2, 'a': 3, 'b': 4, 'c': 5, 'ϵ': 0, '↦': 6, '⇥': 7}

        :param alpha_repr: the input source--it may be a dictionary that maps chars to codes,
                        a nested list, a plain string, or the path of a TSV file.
        :type alpha_repr: Union[str, dict, list]

        :param unknown_class_representant: any character in this string should map to the code for 'unknown', as well as all members of its CharClass class.
        :type unknown_class_representant: str
        """

        self._utf_2_code = {}
        if type(alpha_repr) is dict:
            # in case the input dictionary already contains the special symbols 
            cleaned = { s:c for s,c in alpha_repr.items() if s not in (self.null_symbol, 
                                                                       self.start_of_seq_symbol, 
                                                                       self.end_of_seq_symbol) }
            self._utf_2_code = self.from_dict( cleaned )

        elif type(alpha_repr) is str or isinstance(alpha_repr, Path):
            alpha_path = Path( alpha_repr ) if type(alpha_repr) is str else alpha_repr
            if alpha_path.suffix == '.tsv' and alpha_path.exists():
                #print("__init__( tsv_path )")
                self._utf_2_code = self.from_tsv( alpha_repr )  
            else:
                self._utf_2_code = self.from_string( alpha_repr )
        elif type(alpha_repr) is list:
            self._utf_2_code = self.from_list( alpha_repr )

        # find code of unknown/uninterpretable chars and map it its predefined symbol 
        self.unknown_class_representant = unknown_class_representant if unknown_class_representant else ''

        self.finalize()

        # crude, character-splitting function makes do for now
        # TODO: a proper tokenizer that splits along the given alphabet
        self.tokenize = self.tokenize_crude if tokenizer is None else tokenizer

    @property
    def many_to_one( self ):
        return not all(i==1 for i in Counter(self._utf_2_code.values()).values())



    def finalize( self ) -> None:
        """Finalize the alphabet's data:

        * Add virtual symbols: EOS, SOS, null symbol
        * compute the reverse dictionary
        """
        self._utf_2_code[ self.null_symbol ] = self.null_value
        for s in (self.start_of_seq_symbol, self.end_of_seq_symbol):
            if s not in self._utf_2_code:
                self._utf_2_code[ s ] = self.maxcode+1
        
        if self.many_to_one:
            self._code_2_utf = { c:s.lower() for (s,c) in sorted(self._utf_2_code.items(), reverse=True) }
        else:
            self._code_2_utf = { c:s for (s,c) in sorted(self._utf_2_code.items(), reverse=True) }
            
        self.default_symbol, self.default_code = self.null_symbol, self.null_value

        if self.unknown_class_representant:
            cr_code = self.get_code( self.unknown_class_representant )
            if cr_code is not None:
                self._code_2_utf[ cr_code ] = self.unknown_symbol_utf

    def to_tsv( self, filename: Union[str,Path]) -> None:
        """ Dump to TSV file.
        
        :param filename: path to TSV file
        :type filename: Union[str,Path]
        """
        with open( filename, 'w') as of:
            print(self, file=of)



    def to_list( self, exclude: list=[] )-> List[Union[str,list]]:
        """ Return a list representation of the alphabet. 

        Virtual symbols (EoS, SoS, null) are not included, so that it can be fed back
        to the initialization method.

        :param exclude: list of symbols that should not be included into the resulting list.
        :type exclude: List[str]

        >>> alphabet.Alphabet([['a','A'],'b','c']).to_list(['a','b'])
        ['A', 'c']

        :returns: a list of lists or strings.
        :rtype: List[Union[str,list]]
        """
        code_2_utfs = {}
        for (s,c) in self._utf_2_code.items():
            if s in (self.start_of_seq_symbol, self.end_of_seq_symbol, self.null_symbol) or s in exclude:
                continue
            if c in code_2_utfs:
                code_2_utfs[c].add( s )
            else:
                code_2_utfs[c]=set([s])
        return sorted([ sorted(list(l)) if len(l)>1 else list(l)[0] for l in code_2_utfs.values() ], key=lambda x: x[0])
        

    @classmethod
    def from_tsv(cls, tsv_filename: str, prototype=False) -> Dict[str,int]:
        """ Initialize an alphabet dictionary from a TSV file.

        Assumption: if it is not a prototype, the TSV file always contains a correct mapping,
        but the symbols need to be sorted before building the dictionary, to ensure a 
        deterministic mapping of codes to symbols; if it is a prototype, the last column in each
        line is -1 (a dummy for the code) and the previous columns store the symbols that should
        map to the same code.

        :param tsv_filename: a TSV file of the form::

            <symbol>     <code>
        :type tsv_filename: str
        :param prototype: if True, the TSV file may store more than 1 symbol on the same
                              line, as well as a proto-code at the end (-1); codes are to be generated.
        :type prototype: bool

        :returns: a dictionary `{ <symbol>: <code> }`
        :rtype: Dict[str, int]
        """
        with open( tsv_filename, 'r') as infile:
            if prototype:
                if next(infile).split('\t')[-1].rstrip() != '-1':
                    raise ValueError("File is not a prototype TSV. Format expected:"
                                     "<char1>    [<char2>,    ...]    -1")
                infile.seek(0)
                # prototype TSV may have more than one symbol on the same line, for many-to-one mapping
                # Building list-of-list from TVS:
                # A   ae   -1          
                # O   o    ö    -1 ---> [['A', 'ae'], 'O', 'o', 'ö'], ... ]
                print("Loading alphabet")
                lol = [ s if len(s)>1 else s[0] for s in [ line.split('\t')[:-1] for line in infile if re.match(r'\s*$', line) is None ]]
                
                return cls.from_list( lol )

            return { s:int(c.rstrip()) for (s,c) in sorted([ line.split('\t') for line in infile if re.match(r'\s*$', line) is None]) }

    @classmethod
    def from_list(cls, symbol_list: List[Union[List,str]]) -> Dict[str,int]:
        """
        Construct a symbol-to-code dictionary from a list of strings or sublists of symbols (for many-to-one alphabets):
        symbols in the same sublist are assigned the same label.

        Works on many-to-one, compound symbols. Eg. 

        >>> from_list( [['A','ae'], 'b', ['ü', 'ue', 'u', 'U'], 'c'] )
        { 'A':1, 'U':2, 'ae':1, 'b':3, 'c':4, 'u':5, 'ue':5, ... }

        :param symbol_list: a list of either symbols (possibly with more than one characters) or sublists of symbols that should map to the same code.
        :type symbol_list: List[Union[List,str]]

        :returns: a dictionary mapping symbols to codes.
        :rtype: Dict[str,int]
        """

        # if list is not nested (one-to-one)
        if all( type(elt) is str for elt in symbol_list ):
            return {s:c for (c,s) in enumerate( sorted( symbol_list), start=1)}

        # nested list (many-to-one)
        def sort_and_label( lol ):
            # remove SoS and EoS symbols first
            lol = [ elt for elt in lol if type(elt) is list or (elt.lower() != cls.start_of_seq_symbol and elt.lower() != cls.end_of_seq_symbol) ]
            return [ (c,s) for (c,s) in enumerate(sorted([ sorted(sub) for sub in lol ], key=lambda x: x[0]), start=1)]

        alphadict =dict( sorted( { s:c for (c,item) in sort_and_label( symbol_list ) for s in item if not s.isspace() or s==' ' }.items()) ) 
        return alphadict

    @classmethod
    def from_dict(cls, mapping: Dict[str,int]) -> Dict[str,int]:
        """ Construct an alphabet from a dictionary. The input dictionary need not be sorted.

        :param mapping:
            a dictionary of the form `{ <symbol>: <code> }`; a symbol may have one or more characters.
        :type mapping: Dict[str,int]

        :returns: a sorted dictionary.
        :rtype: Dict[str,int]
        """
        alphadict = dict(sorted(mapping.items()))
        return alphadict

    @classmethod
    def from_string(cls, stg: str ) -> Dict[str,int]:
        """ Construct a one-to-one alphabet from a single string.

        :param stg: a string of characters.
        :type stg: str

        :returns: a `{ code: symbol }` mapping.
        :rtype: Dict[str,int]
        """
        alphadict = { s:c for (c,s) in enumerate(sorted(set( [ s for s in stg if not s.isspace() or s==' ' ])), start=1) }
        return alphadict
        
    @classmethod
    def prototype_from_data_paths(cls, 
                                paths: List[str], 
                                merge:List[str]=[],
                                exclude:List[str]=[],
                                unknown:str='') -> Tuple[ Alphabet, Dict[str,str]]:
        """Given a list of GT transcription file paths, return an alphabet.

        :param paths: a list of file path (wildards accepted).
        :type paths: List[str]

        :param merge: 
            for each of the provided subsequences, merge those output sublists that contain the characters
            in it. Eg. `merge=['ij']` will merge the `'i'` sublist (`[iI$î...]`) with the `'j'` sublist (`[jJ...]`)
        :type merge: List[str]

        :param exclude: a list of Alphabet class names to exclude (keys in the Alphabet categories attribute).
        :type exclude: List[str]

        :param unknown: 
            a stand-in for the one class of characters that have to map
            on the 'unknown' code.
        :type unknown: str

        :returns: a pair with
             * an Alphabet object
             * a dictionary `{ symbol: [filepath, ... ]}` that assign to each symbols all the files in which it appears.
        :rtype: Tuple[Alphabet, Dict[str,str]]

        """
        assert type(paths) is list
        charset = set()
        file_paths = []
        for p in paths:
            path = Path(p)
            if '*' in path.name:
                file_paths.extend( path.parent.glob( path.name ))
            elif path.exists():
                file_paths.append( path )

        char_to_file = {}
        for fp in file_paths:
            with open(fp, 'r') as infile:
                chars_in_this_file = set( char for line in infile for char in list(line.strip())  )
                for c in chars_in_this_file:
                    if c in char_to_file:
                        char_to_file[ c ].append( fp.name )
                    else:
                        char_to_file[ c ] = [ fp.name ]
                charset.update( chars_in_this_file )

        charset.difference_update( set( char for char in charset if char.isspace() and char!=' '))    

        weird_chars = set( char for char in charset if not CharClass.in_domain( char ))

        if weird_chars:
            warnings.warn("You may want to double-check the following characters: {}".format( weird_chars ))

        symbol_list = CharClass.build_subsets(charset, exclude=exclude) if many_to_one else sorted(charset)
        
        symbol_list = cls.merge_sublists( symbol_list, merge )

        return ( cls(cls.deep_sorted(symbol_list), unknown_class_representant=unknown), char_to_file)

    @classmethod
    def prototype_from_data_samples(cls, 
                                    transcriptions: List[str], 
                                    merge:List[str]=[], 
                                    exclude:List[str]=[],
                                    unknown:str='') -> Alphabet:
        """ Given a list of GT transcription strings, return an Alphabet.

        :param paths: a list of transcriptions. 
        :type paths: List[str]

        :param merge: 
            for each of the provided subsequences, merge those output sublists that contain the characters
            in it. Eg. `merge=['ij']` will merge the `'i'` sublist (`[iI$î...]`) with the `'j'` sublist (`[jJ...]`)
        :type merge: List[str]

        :param exclude: 
            a list of Alphabet class names to exclude (keys in the Alphabet categories attribute).
        :type exclude: List[str]

        :param unknown: 
            a stand-in for the one class of characters that have to map 
            on the 'unknown' code.
        :type unknown: str
        :returns: an Alphabet object
        :rtype: Alphabet

        """
        charset = set()

        for tr in transcriptions:
            chars = set( list(tr.strip())  )
            charset.update( chars )
        charset.difference_update( set( char for char in charset if char.isspace() and char!=' '))    

        non_ascii_chars = set( char for char in charset if ord(char)>127 )
        weird_chars = set( char for char in non_ascii_chars if not CharClass.in_domain( char ))
        non_ascii_chars.difference_update( weird_chars )

        if non_ascii_chars:
            warnings.warn("The following characters are not in the ASCII set but look like reasonable Unicode symbols: {}".format( non_ascii_chars ))
        if weird_chars:
            warnings.warn("You may want to double-check the following characters: {}".format( weird_chars ))


        symbol_list = CharClass.build_subsets(charset, exclude=exclude) if many_to_one else sorted(charset)
        
        symbol_list = cls.merge_sublists( symbol_list, merge )        

        return cls( cls.deep_sorted(symbol_list), unknown_class_representant=unknown)



        
    @classmethod
    def prototype_from_scratch(cls, 
                                merge:List[str]=[], 
                                exclude:List[str]=[],
                                unknown:str='') -> Alphabet:
        """ Build a tentative, "universal", alphabet from scratch, without regard to the data: it 
        maps classes of characters to common code, as described in the CharacterClass below.
        The resulting encoding is rather short and lends itself to a variety of datasets.
        The output can be redirected on file, reworked and then fed back through `from_tsv()`.

        :param merge: 
            for each of the provided subsequences, merge those output sublists that contain the characters
            in it. Eg. `merge=['ij']` will merge the `'i'` sublist (`[iI$î...]`) with the `'j'` sublist (`[jJ...]`)
        :type merge: List[str]

        :param exclude: 
            a list of Alphabet class names to exclude (keys in the Alphabet categories
            attribute).
        :type exclude: List[str]

        :param unknown: 
            a stand-in for the one class of characters that have to map on the 
            'unknown' code.
        :type unknown: str

        :returns: an Alphabet object
        :rtype: Alphabet

        """

        symbol_list = CharClass.build_subsets( exclude=exclude )
        symbol_list = cls.merge_sublists( symbol_list, merge )        

        return cls(cls.deep_sorted(symbol_list), unknown_class_representant=unknown)

        
    def __len__( self ):
        return len( self._code_2_utf )

    def __str__( self ) -> str:
        """ A summary
        """
        one_symbol_per_line = '\n'.join( [ f'{s}\t{c}' for (s,c) in  sorted(self._utf_2_code.items()) ] )
        return one_symbol_per_line.replace( self.null_symbol, '\u03f5' )

    def __repr__( self ) -> str:
        return repr( self._utf_2_code )


    @property
    def maxcode( self ):
        #print(self._code_2_utf.keys())
        return max( list(self._utf_2_code.values()) )
    
    def __eq__( self, other ):
        return self._utf_2_code == other._utf_2_code

    def __contains__(self, v ):
        if type(v) is str:
            return (v in self._utf_2_code)
        if type(v) is int:
            return (v in self._code_2_utf)
        return False

    def __getitem__( self, i: Union[int,str]) -> Union[int,str]:
        if type(i) is str:
            return self._utf_2_code[i]
        if type(i) is int:
            return self._code_2_utf[i]

    def get_symbol( self, code, all=False ) -> Union[str, List[str]]:
        """ Return the class representant (default) or all symbols that map on the given code.

        :param code: a integer code.
        :type code: int
        
        :param all: 
            if True, returns all symbols that map to the given code; if False (default),
            returns the class representant.
        :type all: bool

        :returns: the default symbol for this code, or the list of matching symbols.
        :rtype: Union[str, List[str]]
        """
        if all:
            return [ s for (s,c) in self._utf_2_code.items() if c==code ]
        return self._code_2_utf[ code ] if code in self._code_2_utf else self.default_symbol


    def get_code( self, symbol ) -> int:
        """Return the code on which the given symbol maps.

        For symbols that are not in the alphabet, the default code (null) is returned.

        :param symbol: a character.
        :type symbol: str

        :returns: an integer code
        :rtype: int
        """
        return self._utf_2_code[ symbol ] if symbol in self._utf_2_code else self.default_code
    

    def stats( self ) -> dict:
        """ Basic statistics.
        """
        return { 'symbols': len(set(self._utf_2_code.values()))-3,
                 'codes': len(set(self._utf_2_code.keys()))-3,
               }


    def symbol_intersection( self, alpha: Self )->set:
        """ Returns a set of those symbols that can be encoded in both alphabets.

        :param alpha: an Alphabet object.
        :type alpha: Alphabet

        :returns: a set of symbols.
        :rtype: set
        """
        return set( self._utf_2_code.keys()).intersection( set( alpha._utf_2_code.keys()))

    def symbol_differences( self, alpha: Self ) -> Tuple[set,set]:
        """ Compute the differences of two alphabets.

        :param alpha: an Alphabet object.
        :type alpha: Alphabet

        :returns: a tuple with two sets - those symbols that can be encoded with the first alphabet, but
                  not the second one; and conversely.
        :rtype: Tuple[set, set]
        """
        return ( set(self._utf_2_code.keys()).difference( set( alpha._utf_2_code.keys())),
                 set(alpha._utf_2_code.keys()).difference( set( self._utf_2_code.keys())))

    def add_symbols( self, symbols ):
        """ Add one or more symbol to the alphabet.

        :param symbols:
                a list whose elements can be individual chars or list of chars that should map 
                to the same code. A symbol (or group of symbols) that should be merged with an
                existing group should be given a a list that comprise the new symbol(s) and any
                symbol that already belong to the alphabet's group.
        :type symbols: List[list,str]

        :returns: the alphabet.
        :rtype: Alphabet
        """
        def argfind(lst, x):
            for i,elt in enumerate(lst):
                if x == elt:
                    return i
                if type(elt) is list and x in elt:
                    return i
            return None
        
        if len(symbols) == 0:
            return self

        list_form = self.to_list()
        for addition in symbols:
            if type(addition) is not list and addition not in self:
                list_form.append( addition )
            else:
                hooks = [ s for s in addition if s in self ] 
                if len(hooks) == 0:
                    list_form.append( addition )
                elif len(hooks) > 1 and len(list(itertools.groupby( [ self.get_code(h) for h in hooks ]))) > 1:
                    raise ValueError("Merging distinct symbol groups is not allowed: check that that provided hooks are valid.")
                else:
                    to_merge = list(set( addition ) - set( hooks ))
                    # find index to merge in
                    i = argfind( list_form, hooks[0] )
                    if type(list_form[i]) is list:
                        list_form[i].extend( to_merge )
                    else:
                        list_form[i] = [list_form[i]] + to_merge 

        self._utf_2_code = self.from_list( list_form )

        self.finalize()
        return self

    def remove_symbols( self, symbol_list: list ):
        """ Suppress one or more symbol from the alphabet.

        The list format is used here as a convenient intermediate representation.

        :param symbol_list: a list of symbols to be removed from the mapping.
        :type symbol_list: list

        :returns: the alphabet itself.
        :rtype: Alphabet
        """
        self._utf_2_code = self.from_list( self.to_list( exclude=symbol_list))
        self.finalize()
        return self

    def remove_symbol_class( self, symbol_class: str ):
        """ Suppress a class of symbols from the alphabet.

        :param symbol_class: a key in the CharacterClass' character_classes dictionary.
        :type symbol_list: list

        :returns: the alphabet itself.
        :rtype: Alphabet
        """
        self._utf_2_code = self.from_list( self.to_list( exclude=list( CharClass.get(symbol_class) )))
        self.finalize()
        return self

    def encode(self, sample_s: str, ignore_unknown=False) -> list:
        """Encode a message string with integers: the string is segmented first.

        .. todo::

            flag for handling of unknown characters (ignore or encode as null)

        :param sample_s: message string, clean or not.
        :type sample_s: str

        :param ignore_unknown:
            if True, symbols that are not in the dictionary are ignored. Default
            is False (unknown symbols are mapped to the null value).
        :type ignore_unknown: bool

        :returns: a list of integers; symbols that are not in the alphabet yield
                  a default code while generating a user warning.
        :rtype: list
        """
        sample_s = self.normalize_spaces( sample_s )
        return [ self.get_code( t ) for t in self.tokenize( sample_s ) ]

    def encode_one_hot( self, sample_s: List[str]) -> Tensor:
        """ 
        One-hot encoding of a message string.
        """
        encode_int = self.encode( sample_s )
        return torch.tensor([[ 0 if i!=c else 1 for i in range(len(self)) ] for c in encode_int ],
                dtype=torch.bool)

    def encode_batch(self, samples_s: List[str], padded=True, ignore_unknown=False ) -> Tuple[Tensor, Tensor]:
        """ Encode a batch of messages.

        .. todo::

            flag for handling of unknown characters (ignore or encode as null)

        :param samples_s: a list of strings
        :type samples_s: List[str]

        :param padded: 
                if True (default), return a tensor of size (N,S) where S is the maximum 
                length of a sample mesg; otherwise, return an unpadded 1D-sequence of labels.
        :type padded: bool

        :param ignore_unknown:
            if True, symbols that are not in the dictionary are ignored. Default
            is False (unknown symbols are mapped to the null value).
        :type ignore_unknown: bool

        :returns: a pair of tensors, with encoded batch as first element
                  and lengths as second element.
        :rtype: Tuple[Tensor, Tensor]
        """
        encoded_samples = [ self.encode( s ) for s in samples_s ]
        lengths = [ len(s) for s in encoded_samples ] 

        if padded:
            batch_bw = torch.zeros( [len(samples_s), max(lengths)], dtype=torch.int64 )
            for r,s in enumerate(encoded_samples):
                batch_bw[r,:len(s)] = torch.tensor( encoded_samples[r], dtype=torch.int64 )
            return (batch_bw, torch.tensor( lengths ))

        concatenated_samples = list(itertools.chain( *encoded_samples ))
        return ( torch.tensor( concatenated_samples ), torch.tensor(lengths))


    def decode(self, sample_t: Tensor, length: int=-1 ) -> str:
        """ Decode an integer-encoded sample.

        :param sample_t: a tensor of integers (W,).
        :type sample_t: Tensor

        :param length: sample's length; if -1 (default), all symbols are decoded.
        :type length: int

        :returns: a string of symbols
        :rtype: str
        """
        length = len(sample_t) if length < 0 else length
        return "".join( [self.get_symbol( c ) for c in sample_t.tolist()[:length] ] )


    def decode_batch(self, samples_nw: Tensor, lengths: Tensor=None ) -> List[ str ]:
        """ Decode a batch of integer-encoded samples.

        
        :param sample_nw: each row of integers encodes a string.
        :type sample_nw: Tensor (N,W)

        :param lengths: length to be decoded in each sample; the default is full-length decoding.
        :type lenghts: Tensor (N)

        :returns: a sequence of strings.
        :rtype: list
        """
        if lengths == None:
            sample_count, max_length = samples_nw.shape
            lengths = torch.full( (sample_count,), max_length )
        return [ self.decode( s, lgth ) for (s,lgth) in zip( samples_nw, lengths ) ]


    def decode_ctc(self, msg: np.ndarray ):
        """ Decode the output labels of a CTC-trained network into a human-readable string.
        
        .. code-block::

            >>> alphabet.Alphabet('Hello').decode_ctc(np.array([1,1,0,2,2,2,0,0,3,3,0,3,0,4]))
            'Hello'

        :param msg: a sequence of labels, possibly with duplicates and null values.
        :type msg: np.ndarray
        
        :returns: a string of characters.
        :rtype: str
        """
        # keep track of positions to keep
        keep_idx = np.zeros( msg.shape, dtype='bool') 
        if msg.size == 0:
            return ''
        # quick removal of duplicated values
        keep_idx[0] = msg[0] != self.null_value 
        keep_idx[1:] = msg[1:] != msg[:-1] 
        # removal of null chars
        keep_idx = np.logical_and( keep_idx, msg != self.null_value )

        return ''.join( self.get_symbol( c ) for c in msg[ keep_idx ] )
        

    @staticmethod
    def deep_sorted(list_of_lists: List[Union[str,list]]) ->List[Union[str,list]]:
        """ Sort a list that contains either lists of strings, or plain strings.

        Eg.

        .. code-block::

            >>> deep_sorted(['a', ['B', 'b'], 'c', 'd', ['e', 'E'], 'f'])
            [['B', 'b'], ['E', 'e'], 'a', 'c', 'd', 'f']
        

        """
        return sorted([sorted(i) if len(i)>1 else i for i in list_of_lists],
                       key=lambda x: x[0])


    def tokenize_crude( self, mesg: str, quiet=True ) -> List[str]:
        """ Tokenize a string into tokens that are consistent with the provided alphabet.
        A very crude splitting, as a provision for a proper tokenizer. Spaces
        are normalized (only standard spaces - ``' '=\\u0020``)), with duplicate spaces removed.

        :param mesg: a string
        :type mesg: str

        :returns: a list of characters.
        :rtype: List[str]
        """
        if not quiet:
            missing = set( s for s in mesg if s not in self )
            if len(missing)>0:
                warnings.warn('The following chars are not in the alphabet: {}'\
                          ' →  code defaults to {}'.format( [ f"'{c}'={ord(c)}" for c in missing ], self.default_code ))

        return list( mesg )

    @staticmethod
    def normalize_spaces(mesg: str) -> str:
        """ Normalize the spaces: 

        * remove trailing spaces
        * all spaces mapped to standard space (``' '=\\u0020``)
        * duplicate spaces removed

        Eg.

        .. code-block::
             
            >>> normalize_spaces('\\t \\u000Ba\\u000C\\u000Db\\u0085c\\u00A0\\u2000\\u2001d\\u2008\\u2009e')
            ['a b c d e']


        :param mesg: a string
        :type mesg: str 

        :returns: a string
        :rtype: str
        """
        return re.sub( r'\s+', ' ', mesg.strip())
        

    @staticmethod
    def merge_sublists( symbol_list: List[Union[str,list]], merge:List[str]=[] ) -> List[Union[str,list]]:
        """
        Given a nested list and a list of strings, merge the lists contained in <symbol_list>
        such that characters joined in a <merge> string are stored in the same list.

        :param merge: 
            for each of the provided subsequences, merge those output sublists that contain the characters
            in it. Eg. ``merge=['ij']`` will merge the ``'i'`` sublist (``[iI$î...]``) with the ``'j'`` sublist (``[jJ...]``)
        :type merge: List[str]

        :returns: a list of lists.
        """
        if not merge:
            return symbol_list

        symbol_list = symbol_list.copy()

        to_delete = []
        to_add = []
        for mgs in merge:
            merged = set()
            for charlist in symbol_list:
                if set(charlist).intersection( set(mgs) ):
                    merged.update( charlist )
                    to_delete.append( charlist )
            if len(merged):
                to_add.append( list(merged) )
        for deleted_subset in to_delete:
            try:
                symbol_list.remove( deleted_subset )
            except ValueError:
                print(f'Could not delete element {deleted_subset} from list of symbols.')
        if len(to_add):
            symbol_list.extend( to_add ) 
        return symbol_list




class CharClass():
    """
    Those character classes should be make it easier to deal with exotic characters
    at different stages of an HTR pipeline:

    * in the pre-processing stage: filter textual unicode-rich transcription
      data by merging, removing, or replacing entire classes of characters, without having
      to hard-code the in the script.

    * in the model+alphabet building stage: build prototype alphabets and many-to-one
      mappings, where subsets of characters can be used as such, or merged at will.
    
    Most of the dictionary's keys consist of a single symbol that serves as a stand-in
    for its class, and is typically used for backward (code-to-symbol) mapping during
    the decoding phase. However, because some classes need both a more explicit key (eg. abbreviations),
    there is a need to distinguish the key function and the stand-in function, 
    by using the first elt of each value to set the class representant.
    Eg. ``'g': ('g', 'gĝğġģḡ')`` means that 'g' is both the key for the 
    set `'gĝğġģḡ'`, as well as its stand-in or class representant.
    ``'Parenthesis': ('|', '()[]/\\|')`` means that the set "Parenthesis" 
    has the ``'|'`` symbol as a stand-in.

    Notes about the categories:
    
    * Capital and lowercase letters in distinct categories (a consumer script
      may merge them later);
    * Recognizable subscript marks ('a', 'o', ...) belong to their 'headletter'
      class (eg. subscript a '\u0363' belongs to the lowercase 'a' class);
    * One-symbol (= 1 Unicode) abbreviations that are clearly anchored on a
      recognizable letter (eg. 'ꝗ','ꝓ', 'ꝑ', ...) belong to that letters'
      character class;
    * Marks or abbrevations whose expansion depends on the context (eg. subscript
      bar =' ̄', 'ƺ', 'ꝰ', 'Ꝯ', ...) are in a category of their own: a HTR pipeline
      consuming data that contain them may chose to ignore them, or include them
      in the alphabet;
    """
     
    character_classes = {
        ' ': (' ', ' '),
        '0': ('0','0'), '1': ('1','1'), '2': ('2','2'), '3': ('3','3'), '4': ('4','4'), 
        '5': ('5','5'), '6': ('6','6'), '7': ('7','7'), '8': ('8','8'), '9': ('9','9'),
        'A': ('A', 'AÁÂÃÄÅÆĂĄÀ'),
        'a': ('a', 'aáâãäåæāăąàæ\u0363'), # subscript 'a'
        'B': ('B', 'B'),
        'b': ('b', 'b'),
        'C': ('C', 'CÇĆĈĊČ'),
        'c': ('c', 'cçćĉċč\u0368'), # subscript 'c'
        'D': ('D', 'DÐĎĐ'),
        'd': ('d', 'dðďđď\u0369'), # subscript 'd'
        'E': ('E', 'EÈÉÊËĒĔĖĘĚ'),
        'e': ('e', 'eèéêëēĕėęě\u0364'), # subscript 'e'
        'F': ('F', 'F'),
        'f': ('f', 'f'),
        'G': ('G', 'GĜĞĠĢ'),
        'g': ('g', 'gĝğġģḡ'),
        'H': ('H', 'HĤĦ'),
        'h': ('h', 'hĥħ\u036a'), # subscript 'h'
        'I': ('I', 'IÌÍÎÏĨĪĬĮİĲ'),
        'i': ('i', 'iìíîïĩīĭįıĳ\u0365'), # subscript 'i'
        'J': ('J', 'JĴ'),
        'j': ('j', 'jĵɉ'),
        'K': ('K', 'KĶ'),
        'k': ('k', 'kķĸ'),
        'L': ('L', 'LĹĻĽĿŁ£'),
        'l': ('l', 'lĺļľŀł'),
        'M': ('M', 'M'),
        'm': ('m', 'm\u036b'), # subscript 'm'
        'N': ('N', 'NÑŃŅŇŊ'),
        'n': ('n', 'nñńņňŉŋ'),
        'O': ('O', 'OÒÓÔÕÖŌŎŐŒ'),
        'o': ('o', 'oòóôõöōŏőœ°\u0366'), # subscript 'o' 
        'P': ('P', 'P'),
        'p': ('p', 'pꝑꝓ'),
        'Q': ('Q', 'Q'),
        'q': ('q', 'qꝗꝙ'),
        'R': ('R', 'RŔŖŘ'),
        'r': ('r', 'rŕŗřˀ\u036c'), # subscript 'r'
        'S': ('S', 'SŚŜŞŠß'),
        's': ('s', 'sśŝşš'),
        'T': ('T', 'TŢŤŦ'),
        't': ('t', 'tţťŧꝷ\u036d'), # subscript 't'
        'U': ('U', 'UÙÚÛÜŨŪŬŮŰŲ'),
        'u': ('u', 'uùúûüũūŭůűų\u0367'), # subscript 'u'
        'V': ('V', 'V'),
        'v': ('v', 'vꝟ\u036e'), # subscript 'v'
        'W': ('W', 'WŴ'),
        'w': ('w', 'wŵ'),
        'X': ('X', 'X'),
        'x': ('x', 'x\u036f'), # subscript 'x'
        'Y': ('Y', 'YŶŸ'),
        'y': ('y', 'yýÿŷ'),
        'Z': ('Z', 'ZŹŻŽ'),
        'z': ('z', 'zźżž'),
        '.': ('.', '.'),
        ',': (',', ':;,✳'),
        '-': ('-', '-¬—='),
        'Diacritic': ('^', "'ʼ" + ''.join([ chr(c) for c in range(0x300,0x316) ])), # variety of diacritics
        'Parenthesis': ('|', '()[]/\\|'),
        'Abbreviation': ('', 'ƺꝙꝮꝯꝫȝꝝ₰ꝛꝰꝭ&¶§₎כּ'),
        'Hebrew': ('א', ''.join([ chr(c) for c in range(0x0591,0x05f5) ])),
    }

    @classmethod
    def is_ascii(cls, char: str) -> bool:
        return ord(char) <= 127

    @classmethod
    def in_domain(cls, char: str) -> bool:
        """ Check that a symbol is in the dictionary."""
        return cls.get_key( char ) is not None

    @classmethod
    def get_class(cls, classname:str) -> str:
        """Get characters in the given class.

        :param classname: a key in the dictionary.
        :type classname: str
        """
        return cls.character_classes[classname][1] if classname in cls.character_classes else None

    @classmethod
    def get_key(cls, char: str ) -> str:
        """ Get key ("head character") for given character 

        :param char: a UTF character. Eg. `'Ä'`
        :type char: str

        :returns: a UTF character, i.e. head-character for its category. Eg. `'a'`
        :rtype: str
        """
        for (k, cat) in cls.character_classes.items():
            if char in cat[1]:
                return k
        return None
    
    @classmethod
    def build_subsets(cls, chars: set = None, exclude=[]) -> List[Union[List,str]]:
        """ From a set of chars, return a list of lists, where each sublist matches one
        of the categories above.


        :param chars: set of individual chars.
        :type chars: set

        :param exclude: names (keys) for those classes of characters that should be included in the output.
        :type exclude: List[str]

        :returns: a list of individual chars or list of chars.
        :rtype: List[Union[List,str]]

        .. code-block::

            >>> build_subsets({'$', 'Q', 'Ô', 'ß', 'á', 'ç', 'ï', 'ô', 'õ', 'ā', 'Ă', 'ķ', 'ĸ', 'ś'})
            [['Ă', 'ā', 'á'], 'ç', 'ï', ['ĸ', 'ķ'], ['Ô', 'õ', 'ô'], 'Q', ['ś', 'ß'], '$'] 

        """
        #{ self.get_key(),c for c in chars }
        chardict = { k:set()  for k in cls.character_classes.keys() }
        lone_chars = []

        # simply returns the classes as a list of list
        if chars is None:
            return [ list(v[1]) if len(v[1])>1 else v[1] for (k,v) in cls.character_classes.items() if k not in exclude ]

        for (k,c) in [ (cls.get_key(c),c) for c in chars ]:
            if k is None:
                lone_chars.append(c)
            elif k not in exclude:
                chardict[k].add(c)
        return [ list(s) if len(s) > 1 else list(s)[0] for s in chardict.values() if len(s) ] + lone_chars




def dummy():
    return True
