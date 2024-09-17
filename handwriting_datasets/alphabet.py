
from typing import Union,Tuple,List
import torch
from torch import Tensor
from pathlib import Path
import warnings




class Alphabet:

    """
    Internally stored as 2 (synchronized) dictionaries
    - one with int code as key and utf char as value.
    - one with utf char as key and int code as value.

    Features:
        - loads from tsv or string
        - white space char (U0020=32)
        - null character (U2205)
        - default character for encoding (when dealing with unknown char)
        - many-to-one code2utf always return the same code (the last): assume that at creation time,
        symbols have been sorted s.t.
        { 1: A, ..., 10: a, ... } -> {
        - compound symbols


    """
    null_symbol = '\u2205'
    start_of_seq_symbol = 'SoS'
    end_of_seq_symbol = 'EoS'


    def __init__( self, alpha_repr: Union[str,dict]=''):

        self._utf_2_code = {}
        if type(alpha_repr) is dict:
            self._utf_2_code = self.from_dict( alpha_repr )
        elif type(alpha_repr) is str:
            #print("__init__( str )")
            alpha_path = Path( alpha_repr )
            if alpha_path.suffix == '.tsv' and alpha_path.exists():
                #print("__init__( tsv_path )")
                self._utf_2_code = self.from_tsv( alpha_repr )  
            else:
                self._utf_2_code = self.from_string( alpha_repr )
        elif type(alpha_repr) is list:
            self._utf_2_code = self.from_list( alpha_repr )

        self.finalize()

    def finalize( self ):
        """
        - Add virtual symbols: EOS, SOS, null symbol
        - compute the reverse dictionary
        """

        self._utf_2_code[ self.null_symbol ] = 0
        for s in (self.start_of_seq_symbol, self.end_of_seq_symbol):
            if s not in self._utf_2_code:
                self._utf_2_code[ s ] = self.maxcode+1
        
        self._code_2_utf = { c:s for (s,c) in self._utf_2_code.items() }
        #print(self._code_2_utf)
        self.default_symbol, self.default_code = self.null_symbol, self._utf_2_code[ self.null_symbol ]


    def remove_symbols( self, symbol_list: list ):
        """
        Suppress one or more symbol from the alphabet. The list format is used as convenient intermediary.

        """
        self._utf_2_code = self.from_list( self.to_list( exclude=symbol_list))
        self.finalize()


    def to_list( self, exclude: list=[] )-> List[Union[str,list]]:
        """
        Return a list representation of the alphabet, minus the virtual symbols, so that
        it can be fed back to the initialization method.

        Input:
            exclude (list): list of symbols that should not be included into the resulting list.
        Output:
            list: a list of list or strings.
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
    def from_tsv(cls, tsv_filename: str, prototype=False) -> dict:
        """
        Assumption: the TSV file always contains a correct mapping, but the symbols may need
        to be sorted before building the dictionary, to ensure a deterministic mapping of
        codes to symbols.

        Input:
            tsv_filename (str): a TSV file of the form
                                <symbol>     <code>
            prototype (bool): if True, the TSV file only contains a proto-codes (-1); codes are
                              to be generated.
        Output:
            dict: { <symbol>: <code> }
        """
        alphadict = {}
        with open( tsv_filename, 'r') as infile:
            if prototype:
                alphadict.update( { s:c for (c,s) in enumerate(sorted([ line.split('\t')[0] for line in infile ])) })
            else:
                alphadict.update( { s:int(c.rstrip()) for (s,c) in sorted([ line.split('\t') for line in infile ]) })
        return alphadict

    @classmethod
    def from_list(cls, symbol_list: List[Union[List,str]]) -> dict:
        """
        Construct a symbol-to-code dictionary from a list of strings or sublists of symbols (for many-to-one alphabets):
        symbols in the same sublist are assigned the same label.

        Works on many-to-one, compound symbols:
        Eg. [['A','ae'], 'b', ['ü', 'ue', 'u', 'U'], 'c'] -> { 'A':1, 'U':2, 'ae':1, 'b':3, 'c':4, 'u':5, 'ue':5, ... }
        """

        # if list is not nested (one-to-one)
        if all( type(elt) is str for elt in symbol_list ):
            return {s:c for (c,s) in enumerate( sorted( symbol_list), start=1)}

        # nested list (many-to-one)
        def sort_and_label( lol ):
            return [ (c,s) for (c,s) in enumerate(sorted([ sorted(sub) for sub in lol ], key=lambda x: x[0]), start=1)]

        alphadict =dict( sorted( { s:c for (c,item) in sort_and_label( symbol_list ) for s in item if not s.isspace() or s==' ' }.items()) ) 
        return alphadict

    @classmethod
    def from_dict(cls, mapping: dict) -> dict:
        alphadict = dict(sorted(mapping.items()))
        return alphadict

    @classmethod
    def from_string(cls, stg: str ) -> dict:
        """
        Output:
            dict[str,int]: a { code: symbol } mapping.
        """
        alphadict = { s:c for (c,s) in enumerate(sorted(set( [ s for s in stg if not s.isspace() or s==' ' ])), start=1) }
        return alphadict
        
    @classmethod
    def prototype(cls, paths: List[str], out_format="list", many_to_one=True) -> Union[str,List[str]]:
        """
        Given a list of GT transcription file paths, return a TSV representation of the alphabet.
        Normally not made for immediate consumption, it allows for a quick look at the character set.
        The output can be redirected on file, reworked and then fed back through `from_tsv()`.

        Args:
            paths (List[str]): a list of file path (wildards accepted).
            out_format (str): if 'list' (the default), output is a Python list, that can be fed to 
                          the from_list() initialization method; if 'tsv', a TSV str, without
                          the virtual symbols.
        Output:
            Union[list,str]: a list of symbols, or the alphabet representation in TSV form
                             (<symbol>   -1) where -1 is a placeholder for the code.

        """
        charset = set()
        file_paths = []
        for p in paths:
            path = Path(p)
            if '*' in path.name:
                file_paths.extend( path.parent.glob( path.name ))
            elif path.exists():
                file_paths.append( path )
        #print(file_paths)

        char_to_file = {}
        for fp in file_paths:
            with open(fp, 'r') as infile:
                chars_in_this_file = set( char for line in infile for char in list(line.strip()) )
                for c in chars_in_this_file:
                    if c in char_to_file:
                        char_to_file[ c ].append( fp.name )
                    else:
                        char_to_file[ c ] = [ fp.name ]
                charset.update( chars_in_this_file )
        #        print(charset)
        charset.difference_update( set( char for char in charset if char.isspace() and char!=' '))    
        non_ascii_chars = [ char for char in charset if ord(char)>127 ]
        if non_ascii_chars:
            warnings.warn("The following characters are not in the ASCII set: {}".format( non_ascii_chars ))
        symbol_list = CharClass.build_subsets(charset) if many_to_one else sorted(charset)
        if out_format == 'tsv':
            print(char_to_file)
            return "\n".join( [f"{s}\t-1" for s in symbol_list ] )
        print('symbol_list', symbol_list)
        return (symbol_list, char_to_file)

        
    def __len__( self ):
        return len( self._code_2_utf )

    def __str__( self ) -> str:
        """ A TSV representation of the alphabet
        """
        one_symbol_per_line = '\n'.join( [ f'{s}\t{c}' for (s,c) in  sorted(self._utf_2_code.items()) ] )
        return one_symbol_per_line.replace( self.null_symbol, '\u2205' )

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

    def get_symbol( self, code ) -> str:
        return self._code_2_utf[ code ] if code in self._code_2_utf else self.default_symbol

    def get_code( self, symbol ) -> int:
        return self._utf_2_code[ symbol ] if symbol in self._utf_2_code else self.default_code
    
    def encode(self, sample_s: str) -> Tensor:
        """ 
        Encode a message string with integers. 

        Input:
            sample_s (str): message string; assume clean sample: no newlines nor tabs.

        Output:
            Tensor: a tensor of integers; symbols that are not in the alphabet yield
                    a default code (=max index) while generating a user warning.
        """
        if [ s for s in sample_s if s in '\n\t' ]:
            raise ValueError("Sample contains illegal symbols: check for tabs and newlines chars.")
        missing = [ s for s in sample_s if s not in self ]
        if missing:
                warnings.warn('The following chars are not in the alphabet: {}'\
                          ' →  code defaults to {}'.format( missing, self.default_code ))
        return torch.tensor([ self.get_code( s ) for s in sample_s ], dtype=torch.int64 )

    def encode_one_hot( self, sample_s: List[str]) -> Tensor:
        """ 
        One-hot encoding of a message string.
        """
        encode_int = self.encode( sample_s )
        return torch.tensor([[ 0 if i!=c else 1 for i in range(len(self)) ] for c in encode_int ],
                dtype=torch.bool)

    def encode_batch(self, samples_s: List[str] ) -> Tuple[Tensor, Tensor]:
        """
        Encode a batch of messages.

        Input:
            samples_s (list): a list of strings

        Output:
            tuple( Tensor, Tensor ): a pair of tensors, with encoded batch as first element
                                     and lengths as second element.
        """
        lengths = [ len(s) for s in samples_s ] 
        batch_bw = torch.zeros( [len(samples_s), max(lengths)] )
        for r,s in enumerate(samples_s):
            batch_bw[r,:len(s)] = self.encode( s )
        return (batch_bw, torch.tensor( lengths ))


    def decode(self, sample_t: Tensor, length: int=-1 ) -> str:
        """ 
        Decode an integer-encoded sample.

        Input:
            sample_t (Tensor): a tensor of integers.
            length (int): sample's length; if -1 (default), all symbols are decoded.
        Output:
            str: string of symbols.
        """
        length = len(sample_t) if length < 0 else length
        return "".join( [self.get_symbol( c ) for c in sample_t.tolist()[:length] ] )


    def decode_batch(self, samples_bw: Tensor, lengths: Tensor=None ) -> List[ str ]:
        """
        Decode a batch of integer-encoded samples.

        Input:
            sample_bw (Tensor): each row of integer encodes a string.
            lengths (int): length to be decoded in each sample; the default
                           is full-length decoding.

        Output:
            list: a sequence of strings.
        """
        if lengths == None:
            sample_count, max_length = samples_bw.shape
            lengths = torch.full( (sample_count,), max_length )
        return [ self.decode( s, lgth ) for (s,lgth) in zip( samples_bw, lengths ) ]
        
class CharClass():

    character_categories = {
        'a': 'aAáÁâÂãÃäÄåÅæÆāĂăĄą',
        'b': 'bB',
        'c': 'cCçÇĆćĈĉĊċČč',
        'd': 'dDðÐĎďĐđ',
        'e': 'eEèÈéÉêÊëËĒēĔĕĖėĘęĚě',
        'f': 'fF',
        'g': 'gGĜĝĞğĠġĢģ',
        'h': 'hHĤĥĦħ',
        'i': 'iIìÌíÍîÎïÏĨĩĪīĬĭĮįİıĲĳ',
        'j': 'jJĴĵ',
        'k': 'kKĶķĸ',
        'l': 'lLĹĺĻļĽľĿŀŁł',
        'm': 'mM',
        'n': 'nNñÑŃńŅņŇňŉŊŋ',
        'o': 'oOòÒóÓôÔõÕöÖŌōŎŏŐőŒœ',
        'p': 'pP',
        'q': 'qQ',
        'r': 'rRŔŕŖŗŘř',
        's': 'sSŚśŜŝŞşŠšß',
        't': 'tTŢţŤťŦŧ',
        'u': 'uUùÙúÚûÛüÜŨũŪūŬŭŮůŰűŲų',
        'v': 'vV',
        'w': 'wWŴŵ',
        'x': 'xX',
        'y': 'yYŶŷŸ',
        'z': 'zZŹźŻżŽ'
    }

    @classmethod
    def get_key(cls, char: str ):
        """ Get key ("head character") for given character 

        Input:
            char (str): a UTF character. Eg. `'Ä'`

        Output:
            str: a UTF character, i.e. head-character for its category. Eg. `'a'`
        """
        for (k, cat) in cls.character_categories.items():
            if char in cat:
                return k
        return None
    
    @classmethod
    def build_subsets(cls, chars: set) -> List[Union[List,str]]:
        """ From a set of chars, return a list of lists, where
        each list matches one of the categories above.

        Eg. 
        
        ~~~python
        >>> build_subsets({'$', 'Q', 'Ô', 'ß', 'á', 'ç', 'ï', 'ô', 'õ', 'ā', 'Ă', 'ķ', 'ĸ', 'ś'})
        {'$', 'Q', 'Ô', 'ß', 'á', 'ç', 'ï', 'ô', 'õ', 'ā', 'Ă', 'ķ', 'ĸ', 'ś'}
        ~~~
        """
        #{ self.get_key(),c for c in chars }
        chardict = { k:set()  for k in cls.character_categories.keys() }
        lone_chars = []
        for (k,c) in [ (cls.get_key(c),c) for c in chars ]:
            if k is None:
                lone_chars.append(c)
            else:
                chardict[k].add(c)
        return [ list(s) if len(s) > 1 else list(s)[0] for s in chardict.values() if len(s) ] + lone_chars


def dummy():
    return True
