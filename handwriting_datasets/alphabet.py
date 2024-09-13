
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

    TODO:
        SOS-EOS chars
        many-to-one

    """
    null_symbol = '\u2205'
    start_of_seq_symbol = 'SoS'
    end_of_seq_symbol = 'EoS'


    def __init__( self, alpha_repr: Union[str,dict]=''):

        self._utf_2_code = {}
        if type(alpha_repr) is dict:
            self._utf_2_code = self.from_dict( alpha_repr )
        elif type(alpha_repr) is str:
            print("__init__( tsv_path )")
            alpha_path = Path( alpha_repr )
            if alpha_path.suffix == '.tsv' and alpha_path.exists():
                print("__init__( tsv_path )")
                self._utf_2_code = self.from_tsv( alpha_repr )  
            else:
                self._utf_2_code = self.from_string( alpha_repr )
        elif type(alpha_repr) is list:
            self._utf_2_code = self.from_list( alpha_repr )

        self.add_virtual_symbols()

        self._code_2_utf = { c:s for (s,c) in self._utf_2_code.items() }
        print(self._code_2_utf)
        self.default_symbol, self.default_code = self.null_symbol, self._utf_2_code[ self.null_symbol ]


    
    def add_virtual_symbols( self ):
        self._utf_2_code[ self.null_symbol ] = 0
        if self.start_of_seq_symbol not in self._utf_2_code:
            self._utf_2_code[ self.start_of_seq_symbol ] = self.maxcode+1
        if self.end_of_seq_symbol not in self._utf_2_code:
            self._utf_2_code[ self.end_of_seq_symbol ] = self.maxcode+1 
        

    @classmethod
    def from_tsv(cls, tsv_filename: str) -> dict:
        """
        Assumption: the TSV file always contains a correct mapping, but the symbols may need
        to be sorted before building the dictionary, to ensure a deterministic mapping of
        codes to symbols.

        Input:
            tsv_filename (str): a TSV file of the form
                                <symbol>     <code>
        Output:
            dict: { <symbol>: <code }
        """
        alphadict = {}
        with open( tsv_filename, 'r') as infile:
            alphadict.update( { s:int(c.rstrip()) for (s,c) in sorted([ line.split('\t') for line in infile ]) })
        return alphadict

    @classmethod
    def from_list(cls, symbol_list: List[Union[List,str]]) -> dict:
        """
        Construct a symbol-to-code dictionary from a list of strings or sublists of symbols (for many-to-one alphabets):
        symbols in the same sublist are assigned the same label.

        TODO: multi-symbol tokens
        Eg. [['A','ae'], 'b', ['ü', 'ue', 'u', 'U'], 'c']] -> ?
        """
        # ensure deterministic encoding
        # Eg. [['A','a'], 'b', ['ü', 'u', 'U'], 'c']] -> ['Aa', 'Uuü', 'b', 'c']
        plain_strings = sorted([ ''.join( item ) if type(item) is list else item for item in symbol_list ])

        alphadict =dict( sorted( { s:c for (c,item) in enumerate(plain_strings, start=1) for s in sorted(item) if not s.isspace() or s==' ' }.items()) ) 
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
    def prototype(cls, paths: List[str]) -> str:
        """
        Given a list of GT transcription file paths, return a TSV representation of the alphabet.
        Normally not made for immediate consumption, it allows for a quick look at the character set.
        The output can be redirected on file, reworked and then fed back through `from_tsv()`.

        Args:
            paths (List[str]): a list of file path (wildards accepted).
        Output:
            str: alphabet representation in TSV form
                 <symbol>   <code>

        """
        charset = set()
        file_paths = []
        for p in paths:
            path = Path(p)
            if '*' in path.name:
                file_paths.extend( path.parent.glob( path.name ))
            elif path.exists():
                file_paths.append( path )
        print(file_paths)

        for fp in file_paths:
            with open(fp, 'r') as infile:
                charset.update( set( char for line in infile for char in list(line.strip()) ))
                print(charset)
        charset.difference_update( set( char for char in charset if char.isspace() and char!=' '))    
        non_ascii_chars = [ char for char in charset if ord(char)>127 ]
        if non_ascii_chars:
            warnings.warn("The following characters are not in the ASCII set: {}".format( non_ascii_chars ))
        return "\n".join( ["0\t\u2205"] + [f"{s}\t{c}" for (c,s) in enumerate(sorted(charset), start = 1) ] )

        
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
        return torch.tensor([[ 0 if i+1!=c else 1 for i in range(len(self)) ] for c in encode_int ],
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
        

def dummy():
    return True
