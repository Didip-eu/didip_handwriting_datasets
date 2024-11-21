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
    """Creating and handling alphabets.

    + one-to-one or many-to-one alphabet, with deterministic mapping either way;
    + prototyping from reasonable subsets of characters to be grouped
    + a choice of input/output sources: TSV, nested lists, mappings.

    """
    null_symbol = '\u03f5'
    null_value = 0
    start_of_seq_symbol = '\u21A6' # '↦' i.e. '|->'
    end_of_seq_symbol = '\u21E5' # '⇥' i.e. '->|'
    unknown_symbol_utf = '?' 

    def __init__( self, alpha_repr: Union[str,dict,list]='', tokenizer=None, unknown_class_representant:str='ƺ') -> None:
        """Initialize a new Alphabet object. The special characters are added automatically.

            From a dictionary::

                >>> alphabet.Alphabet({'a':1, 'A': 1, 'b': 2, 'c':3})
                {'A': 1, 'a': 1, 'b': 2, 'c': 3, 'ϵ': 0, '↦': 4, '⇥': 5}

            From a TSV path::

                >>> alphabet.Alphabet('alpha.tsv')
                {'A': 1, 'a': 1, 'b': 2, 'c': 3, 'ϵ': 0, '↦': 4, '⇥': 5}

            From a nested list::

                >>> alphabet.Alphabet([['a','A'],'b','c'])
                {'A': 1, 'a': 1, 'b': 2, 'c': 3, 'ϵ': 0, '↦': 4, '⇥': 5}

            From a string of characters (one-to-one)::

                >>> alphabet.Alphabet('aAbc ')
                {' ': 1, 'A': 2, 'a': 3, 'b': 4, 'c': 5, 'ϵ': 0, '↦': 6, '⇥': 7}

            Returns:
                alpha_repr (Union[str, dict, list]): the input source--it may be a dictionary that maps chars to codes,
                       a nested list, a plain string, or the path of a TSV file.

                unknown_class_representant (str): any character in this string should map to the code for 'unknown', as well
                    as all members of its CharClass class.
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
        """Dump to TSV file.

            Args:
                filename (Union[str,Path]): path to TSV file
        """
        with open( filename, 'w') as of:
            print(self, file=of)



    def to_list( self, exclude: list=[] )-> List[Union[str,list]]:
        """Return a list representation of the alphabet.

        Virtual symbols (EoS, SoS, null) are not included, so that it can be fed back
        to the initialization method.

        Args:
            exclude (List[str]): list of symbols that should not be included into the resulting list. Eg::

                >>> alphabet.Alphabet([['a','A'],'b','c']).to_list(['a','b'])
                ['A', 'c']

        Returns:
             List[Union[str,list]]: a list of lists or strings.
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
        """Initialize an alphabet dictionary from a TSV file.

        Assumption: if it is not a prototype, the TSV file always contains a correct mapping,
        but the symbols need to be sorted before building the dictionary, to ensure a
        deterministic mapping of codes to symbols; if it is a prototype, the last column in each
        line is -1 (a dummy for the code) and the previous columns store the symbols that should
        map to the same code.

        Args:
            tsv_filename (str): pathname of a TSV file of the form::

                <symbol>     <code>

            prototype (bool): if True, the TSV file may store more than 1 symbol on the same
                             line, as well as a proto-code at the end (-1); codes are to be generated.

        Returns:
            Dict[str, int]: a dictionary `{ <symbol>: <code> }`
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
        """Construct a symbol-to-code dictionary from a list of strings or sublists of symbols (for many-to-one alphabets):
        symbols in the same sublist are assigned the same label.

        Works on many-to-one, compound symbols. Eg.::

            >>> from_list( [['A','ae'], 'b', ['ü', 'ue', 'u', 'U'], 'c'] )
            { 'A':1, 'U':2, 'ae':1, 'b':3, 'c':4, 'u':5, 'ue':5, ... }

        Args:
            symbol_list (List[Union[List,str]]): a list of either symbols (possibly with more than one characters) or
                sublists of symbols that should map to the same code.

        Returns:
            Dict[str,int]: a dictionary mapping symbols to codes.
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
        """Construct an alphabet from a dictionary. The input dictionary need not be sorted.

        Args:
            mapping (Dict[str,int]): a dictionary of the form `{ <symbol>: <code> }`; a symbol
                may have one or more characters.

        Returns:
            Dict[str,int]: a sorted dictionary.
        """
        alphadict = dict(sorted(mapping.items()))
        return alphadict

    @classmethod
    def from_string(cls, stg: str ) -> Dict[str,int]:
        """Construct a one-to-one alphabet from a single string.

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
                                many_to_one:bool=False,
                                unknown:str='') -> Tuple[ Alphabet, Dict[str,str]]:
        """Given a list of GT transcription file paths, return an alphabet.

        Args:
            paths (List[str]): a list of file path (wildards accepted).
            merge (List[str]): for each of the provided subsequences, merge those output sublists that
                contain the characters in it. Eg. `merge=['ij']` will merge the `'i'` sublist
                (`[iI$î...]`) with the `'j'` sublist (`[jJ...]`)
            exclude (List[str]): a list of Alphabet class names to exclude (keys in the Alphabet
                categories attribute).
            unknown (str): a stand-in for the one class of characters that have to map on
                the 'unknown' code.
            many_to_one (bool): if True (default), builds a many-to-one alphabet, based on the
                Alphabet class' character classes.

        Returns:
            Tuple[Alphabet, Dict[str,str]]: a pair with * an Alphabet object * a dictionary 
                `{ symbol: [filepath, ... ]}` that assigns to each symbols all the files in which it appears.
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
                                    many_to_one:bool=False,
                                    unknown:str='') -> Alphabet:
        """Given a list of GT transcription strings, return an Alphabet.

        Args:
            paths (List[str]): a list of transcriptions.

            merge (List[str]): for each of the provided subsequences, merge those output 
                sublists that contain the characters in it. Eg. `merge=['ij']` will merge the `'i'`
                sublist (`[iI$î...]`) with the `'j'` sublist (`[jJ...]`)

            exclude (List[str]): a list of Alphabet class names to exclude (keys in the Alphabet
                categories attribute).

            many_to_one (bool): if True (default), builds a many-to-one alphabet, based on
                the Alphabet class' character classes.

            unknown (str) a stand-in for the one class of characters that have to map
                on the 'unknown' code.
        
        Returns:
                Alphabet: an Alphabet object

        """
        charset = set()

        for tr in transcriptions:
            chars = set( list(tr.strip())  )
            charset.update( chars )
        charset.difference_update( set( char for char in charset if char.isspace() and char!=' '))    

        symbol_list = CharClass.build_subsets(charset, exclude=exclude) if many_to_one else sorted(charset)
        symbol_list = cls.merge_sublists( symbol_list, merge )        

        return cls( cls.deep_sorted(symbol_list), unknown_class_representant=unknown)



        
    @classmethod
    def prototype_from_scratch(cls, 
                                merge:List[str]=[], 
                                exclude:List[str]=[],
                                unknown:str='') -> Alphabet:
        """Build a tentative, "universal", alphabet from scratch, without regard to the data: it
        maps classes of characters to common code, as described in the CharacterClass below.
        The resulting encoding is rather short and lends itself to a variety of datasets.
        The output can be redirected on file, reworked and then fed back through `from_tsv()`.

        Args:
            merge (List[str]): for each of the provided subsequences, merge those output sublists
                that contain the characters in it. Eg. `merge=['ij']` will merge the `'i'` sublist
                (`[iI$î...]`) with the `'j'` sublist (`[jJ...]`)

            exclude (List[str]): a list of Alphabet class names to exclude (keys in the Alphabet
                categories attribute).

            unknowni (str): a stand-in for the one class of characters that have to map on the
                'unknown' code.

        Returns:
             Alphabet: an Alphabet object
        """

        symbol_list = CharClass.build_subsets( exclude=exclude )
        symbol_list = cls.merge_sublists( symbol_list, merge )        

        return cls(cls.deep_sorted(symbol_list), unknown_class_representant=unknown)

        
    def __len__( self ):
        return len( self._code_2_utf )

    def __str__( self ) -> str:
        """A summary"""
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
        """Return the class representant (default) or all symbols that map on the given code.

        Args:
            code (int): a integer code.

            all (bool): if True, returns all symbols that map to the given code; if False (default),
                returns the class representant.

        Returns:
            Union[str, List[str]]: the default symbol for this code, or the list of matching symbols.
        """
        if all:
            return [ s for (s,c) in self._utf_2_code.items() if c==code ]
        return self._code_2_utf[ code ] if code in self._code_2_utf else self.default_symbol

    def get_code( self, symbol ) -> int:
        """Return the code on which the given symbol maps.

        For symbols that are not in the alphabet, the default code (null) is returned.

        Args:
            symbol (str): a character.

        Returns:
            int: an integer code
        """
        return self._utf_2_code[ symbol ] if symbol in self._utf_2_code else self.default_code


    def stats( self ) -> dict:
        """Basic statistics."""
        return { 'symbols': len(set(self._utf_2_code.values()))-3,
                 'codes': len(set(self._utf_2_code.keys()))-3,
               }


    def symbol_intersection( self, alpha: Self )->set:
        """Returns a set of those symbols that can be encoded in both alphabets.

        Args:
            alpha (Alphabet): an Alphabet object.

        Returns:
            set: a set of symbols.
        """
        return set( self._utf_2_code.keys()).intersection( set( alpha._utf_2_code.keys()))

    def symbol_differences( self, alpha: Self ) -> Tuple[set,set]:
        """Compute the differences of two alphabets.

        Args:
            alpha (Alphabet): an Alphabet object.

        Returns:
            Tuple[set, set]: a tuple with two sets - those symbols that can be encoded with the first alphabet, but
                 not the second one; and conversely.
        """
        return ( set(self._utf_2_code.keys()).difference( set( alpha._utf_2_code.keys())),
                 set(alpha._utf_2_code.keys()).difference( set( self._utf_2_code.keys())))

    def add_symbols( self, symbols ):
        """Add one or more symbol to the alphabet.

        Args:
            symbols (List[list,str]): a list whose elements can be individual chars or list of chars that should map
               to the same code. A symbol (or group of symbols) that should be merged with an
               existing group should be given a a list that comprise the new symbol(s) and any
               symbol that already belong to the alphabet's group.

        Returns:
            Alphabet: the alphabet.
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
        """Suppress one or more symbol from the alphabet.

        The list format is used here as a convenient intermediate representation.

        Args:
            symbol_list (list): a list of symbols to be removed from the mapping.

        Returns:
            Alphabet: the alphabet itself.
        """
        self._utf_2_code = self.from_list( self.to_list( exclude=symbol_list))
        self.finalize()
        return self

    def remove_symbol_class( self, symbol_class: str ):
        """Suppress a class of symbols from the alphabet.

        Args:
            symbol_class (list): a key in the CharacterClass' character_classes dictionary.

        Returns:
            Alphabet: the alphabet itself.
        """
        self._utf_2_code = self.from_list( self.to_list( exclude=list( CharClass.get(symbol_class) )))
        self.finalize()
        return self

    def encode(self, sample_s: str, ignore_unknown=False) -> list:
        """Encode a message string with integers: the string is segmented first.

        Todo:
            flag for handling of unknown characters (ignore or encode as null)

        Args:
            sample_s (str): message string, clean or not.
            ignore_unknown (bool): if True, symbols that are not in the dictionary are
                ignored. Default is False (unknown symbols are mapped to the null value).

        Returns:
            list: a list of integers; symbols that are not in the alphabet yield
                a default code while generating a user warning.
        """
        sample_s = self.normalize_spaces( sample_s )
        return [ self.get_code( t ) for t in self.tokenize( sample_s ) ]

    def encode_one_hot( self, sample_s: List[str]) -> Tensor:
        """One-hot encoding of a message string."""
        encode_int = self.encode( sample_s )
        return torch.tensor([[ 0 if i!=c else 1 for i in range(len(self)) ] for c in encode_int ],
                dtype=torch.bool)

    def encode_batch(self, samples_s: List[str], padded=True, ignore_unknown=False ) -> Tuple[Tensor, Tensor]:
        """Encode a batch of messages.

        Args:
            samples_s (List[str]): a list of strings

            padded (bool): if True (default), return a tensor of size (N,S) where S is the maximum
               length of a sample mesg; otherwise, return an unpadded 1D-sequence of labels.

            ignore_unknown (bool): if True, symbols that are not in the dictionary are ignored. Default
                is False (unknown symbols are mapped to the null value).

        Returns:
                Tuple[Tensor, Tensor]: a pair of tensors, with encoded batch as first element
                 and lengths as second element.
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
        """Decode an integer-encoded sample.
        
        Args:
            sample_t (Tensor): a tensor of integers (W,).

            length (int): sample's length; if -1 (default), all symbols are decoded.

        Returns:
             str: a string of symbols
        """
        length = len(sample_t) if length < 0 else length
        return "".join( [self.get_symbol( c ) for c in sample_t.tolist()[:length] ] )


    def decode_batch(self, samples_nw: Tensor, lengths: Tensor=None ) -> List[ str ]:
        """Decode a batch of integer-encoded samples.

        Args:
            sample_nw (Tensor): each row of integers encodes a string.

            lengths (Tensor): length to be decoded in each sample; the default is full-length decoding.

        Returns:
            list: a sequence of strings.
        """
        if lengths == None:
            sample_count, max_length = samples_nw.shape
            lengths = torch.full( (sample_count,), max_length )
        return [ self.decode( s, lgth ) for (s,lgth) in zip( samples_nw, lengths ) ]


    def decode_ctc(self, msg: np.ndarray ):
        """Decode the output labels of a CTC-trained network into a human-readable string. Eg.::

            >>> alphabet.Alphabet('Hello').decode_ctc(np.array([1,1,0,2,2,2,0,0,3,3,0,3,0,4]))
            'Hello'

        Args:
            msg (np.ndarray): a sequence of labels, possibly with duplicates and null values.

        Returns:
               str: a string of characters.
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
        """Sort a list that contains either lists of strings, or plain strings.
        Eg.::

           >>> deep_sorted(['a', ['B', 'b'], 'c', 'd', ['e', 'E'], 'f'])
           [['B', 'b'], ['E', 'e'], 'a', 'c', 'd', 'f']

        Args:
            list_of_lists (List[Union[str,list]]): a list where each element can be a characters or a 
                list of characters.

        Returns:
            List[Union[str,list]]: a sorted list, where each sublist is sorted and the top sorting 
                key is the either the character or the first element of the list to be sorted.
        """
        return sorted([sorted(i) if len(i)>1 else i for i in list_of_lists],
                       key=lambda x: x[0])


    def tokenize_crude( self, mesg: str, quiet=True ) -> List[str]:
        """Tokenize a string into tokens that are consistent with the provided alphabet.
        A very crude splitting, as a provision for a proper tokenizer. Spaces
        are normalized (only standard spaces - `' '=\\u0020`)), with duplicate spaces removed.

        Args:
            mesg (str): a string

        Returns:
            List[str]: a list of characters.
        """
        if not quiet:
            missing = set( s for s in mesg if s not in self )
            if len(missing)>0:
                warnings.warn('The following chars are not in the alphabet: {}'\
                          ' →  code defaults to {}'.format( [ f"'{c}'={ord(c)}" for c in missing ], self.default_code ))

        return list( mesg )

    @staticmethod
    def normalize_spaces(mesg: str) -> str:
        """Normalize the spaces:

        * remove trailing spaces
        * all spaces mapped to standard space (`' '=\\u0020`)
        * duplicate spaces removed

        Eg.::

           >>> normalize_spaces('\\t \\u000Ba\\u000C\\u000Db\\u0085c\\u00A0\\u2000\\u2001d\\u2008\\u2009e')
           ['a b c d e']

        Args:
            mesg (str): a string

        Returns:
               str: a string
        """
        return re.sub( r'\s+', ' ', mesg.strip())
        

    @staticmethod
    def merge_sublists( symbol_list: List[Union[str,list]], merge:List[str]=[] ) -> List[Union[str,list]]:
        """Given a nested list and a list of strings, merge the lists contained in <symbol_list>
        such that characters joined in a <merge> string are stored in the same list.

        Args:
            merge (List[str]): for each of the provided subsequences, merge those output sublists
                that contain the characters in it. Eg. ``merge=['ij']`` will merge the ``'i'``
                sublist (``[iI$î...]``) with the ``'j'`` sublist (``[jJ...]``)

        Returns:
            List[Union[str,list]]: a list of lists.
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
    """Those character classes should be make it easier to deal with exotic characters
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
    Eg. `'g': ('g', 'gĝğġģḡ')` means that 'g' is both the key for the
    set `'gĝğġģḡ'`, as well as its stand-in or class representant.
    `'Parenthesis': ('|', '()[]/\\|')` means that the set "Parenthesis"
    has the `'|'` symbol as a stand-in.

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
        'a': ('a', 'aáâãäåæāăąàæ'),
        'B': ('B', 'B'),
        'b': ('b', 'b'),
        'C': ('C', 'CÇĆĈĊČ'),
        'c': ('c', 'cçćĉċč'),
        'D': ('D', 'DÐĎĐ'),
        'd': ('d', 'dðďđď'),
        'E': ('E', 'EÈÉÊËĒĔĖĘĚ'),
        'e': ('e', 'eèéêëēĕėęě'),
        'F': ('F', 'F'),
        'f': ('f', 'f'),
        'G': ('G', 'GĜĞĠĢ'),
        'g': ('g', 'gĝğġģḡ'),
        'H': ('H', 'HĤĦ'),
        'h': ('h', 'hĥħ'),
        'I': ('I', 'IÌÍÎÏĨĪĬĮİĲ'),
        'i': ('i', 'iìíîïĩīĭįıĳ'),
        'J': ('J', 'JĴ'),
        'j': ('j', 'jĵɉ'),
        'K': ('K', 'KĶ'),
        'k': ('k', 'kķĸ'),
        'L': ('L', 'LĹĻĽĿŁ£'),
        'l': ('l', 'lĺļľŀł'),
        'M': ('M', 'M'),
        'm': ('m', 'm'),
        'N': ('N', 'NÑŃŅŇŊ'),
        'n': ('n', 'nñńņňŉŋ'),
        'O': ('O', 'OÒÓÔÕÖŌŎŐŒ'),
        'o': ('o', 'oòóôõöōŏőœ°'),
        'P': ('P', 'P'),
        'p': ('p', 'pꝑꝓ'),
        'Q': ('Q', 'Q'),
        'q': ('q', 'qꝗꝙ'),
        'R': ('R', 'RŔŖŘ'),
        'r': ('r', 'rŕŗřˀ'),
        'S': ('S', 'SŚŜŞŠß'),
        's': ('s', 'sśŝşš'),
        'T': ('T', 'TŢŤŦ'),
        't': ('t', 'tţťŧꝷ'),
        'U': ('U', 'UÙÚÛÜŨŪŬŮŰŲ'),
        'u': ('u', 'uùúûüũūŭůűų'), 
        'V': ('V', 'V'),
        'v': ('v', 'vꝟ'), 
        'W': ('W', 'WŴ'),
        'w': ('w', 'wŵ'),
        'X': ('X', 'X'),
        'x': ('x', 'x'), 
        'Y': ('Y', 'YŶŸ'),
        'y': ('y', 'yýÿŷ'),
        'Z': ('Z', 'ZŹŻŽ'),
        'z': ('z', 'zźżž'),
        '.': ('.', '.✳'),
        ',': (',', ':;'),
        '-': ('-', '-¬—='),
        'Subscripts': '\u0363\u0368\u0369\u0364\u036a\u0365\u036b\u0366\u036c\u036d\u0367\u036e\u036f', # respectively: acdehimortuvx
        'Diacritic': ('^', "'ʼ" + ''.join([ chr(c) for c in range(0x300,0x316) ])), # variety of diacritics
        'Parenthesis': ('|', '()[]/\\|'),
        'Abbreviation': ('', 'ƺꝙꝮꝯꝫȝꝝ₰ꝛꝰꝭ&§₎כּ'),
        'Hebrew': ('א', ''.join([ chr(c) for c in range(0x0591,0x05f5) ])),
    }

    @classmethod
    def is_ascii(cls, char: str) -> bool:
        return ord(char) <= 127

    @classmethod
    def in_domain(cls, char: str) -> bool:
        """Check that a symbol is in the dictionary."""
        return cls.get_key( char ) is not None

    @classmethod
    def get_class(cls, classname:str) -> str:
        """Get characters in the given class.

        Args:
            classname (str): a key in the dictionary.
        """
        return cls.character_classes[classname][1] if classname in cls.character_classes else None

    @classmethod
    def get_key(cls, char: str ) -> str:
        """Get key for given character

        Args:
           char (str): a UTF character. Eg. `'Ä'`

        Returns:
            str: UTF character or string that serves as a key for its category. Eg. `'a'`
        """
        for (k, cat) in cls.character_classes.items():
            if char in cat[1]:
                return k
        return None


    @classmethod
    def get_representant(cls, char: str ) -> str:
        """Get "head character" for given character's class.

        Args:
            char (str): a UTF character. Eg. `'Ä'`

        Returns:
            str: a UTF character, i.e. head-character for its category. Eg. `'a'`
        """
        for (k, cat) in cls.character_classes.items():
            if char in cat[1]:
                return cat[0]
        return None


    
    @classmethod
    def build_subsets(cls, chars: set = None, exclude=[]) -> List[Union[List,str]]:
        """From a set of chars, return a list of lists, where each sublist matches one
                of the categories above.

        Args:
            chars (set): set of individual chars.

            exclude (List[str]): names (keys) for those classes of characters that should be included in the output.

        Returns: 
            List[Union[List,str]]: a list of individual chars or list of chars. Eg.::

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
