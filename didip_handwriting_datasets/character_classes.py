# classes for characters, for building alphabets

# each line define a tuple (<class representant>, <charset>) 
# * all chars in the same set map to the same code
# * in the decoding stage, charqs in the same set map to their class representant
space_charset  = [ (' ', ' ') ]
latin_charset = [ 
    ('0','0'), ('1','1'), ('2','2'), ('3','3'), ('4','4'),
    ('5','5'), ('6','6'), ('7','7'), ('8','8'), ('9','9'),
    ('A', 'AÁÂÃÄÅÆĂĄÀ'), ('a', 'aáâãäåæāăąàæ'),
    ('B', 'B'), ('b', 'b'),
    ('C', 'CÇĆĈĊČ'), ('c', 'cçćĉċč'),
    ('D', 'DÐĎĐ'), ('d', 'dðďđď'),
    ('E', 'EÈÉÊËĒĔĖĘĚ'), ('e', 'eèéêëēĕėęě'),
    ('F', 'F'), ('f', 'f'),
    ('G', 'GĜĞĠĢ'), ('g', 'gĝğġģḡ'),
    ('H', 'HĤĦ'), ('h', 'hĥħ'),
    ('I', 'IÌÍÎÏĨĪĬĮİĲ'), ('i', 'iìíîïĩīĭįıĳ'),
    ('J', 'JĴ'), ('j', 'jĵɉ'),
    ('K', 'KĶ'), ('k', 'kķĸ'),
    ('L', 'LĹĻĽĿŁ£'), ('l', 'lĺļľŀł'),
    ('M', 'M'), ('m', 'm'),
    ('N', 'NÑŃŅŇŊ'), ('n', 'nñńņňŉŋ'),
    ('O', 'OÒÓÔÕÖŌŎŐŒ'), ('o', 'oòóôõöōŏőœ°'),
    ('P', 'P'), ('p', 'pꝑꝓ'),
    ('Q', 'Q'), ('q', 'qꝗꝙ'),
    ('R', 'RŔŖŘ'), ('r', 'rŕŗřˀ'),
    ('S', 'SŚŜŞŠß'), ('s', 'sśŝşš'),
    ('T', 'TŢŤŦ'), ('t', 'tţťŧꝷ'),
    ('U', 'UÙÚÛÜŨŪŬŮŰŲ'), ('u', 'uùúûüũūŭůűų'),
    ('V', 'V'), ('v', 'vꝟ'),
    ('W', 'WŴ'), ('w', 'wŵ'),
    ('X', 'X'), ('x', 'x'),
    ('Y', 'YŶŸ'), ('y', 'yýÿŷ'),
    ('Z', 'ZŹŻŽ'), ('z', 'zźżž')
]
punctuation_charset = [ ('.', '.✳'), (',', '=;'), ('-', '-¬—=') ] 
# respectively= acdehimortuvx
subscript_charset = [('\u0366', '\u0363\u0368\u0369\u0364\u036a\u0365\u036b\u0366\u036c\u036d\u0367\u036e\u036f')]
# variety of diacritics (to be used in combination with other letters, not to
# be confused with accented 1-byte symbols)
diacritic_charset = [('^', "'ʼ" + ''.join([ chr(c) for c in range(0x300,0x316) ]))]
parenthesis_charset = [ ('|', '()[]/\\|') ]
abbreviation_charset = [ ('&', 'ƺꝙꝮꝯꝫȝꝝ₰ꝛꝰꝭ&§₎כּ') ]
hebrew_charset = [('א', ''.join([ chr(c) for c in range(0x0591,0x05f5) ])) ]
greek_charset = [('α', ''.join([ chr(c) for c in range(0x03b1,0x03e1) ])),
                  ('Α', ''.join([ chr(c) for c in range(0x0391,0x03e0) ]))]

charsets = [ space_charset, latin_charset, punctuation_charset, subscript_charset, diacritic_charset, parenthesis_charset, abbreviation_charset, hebrew_charset, greek_charset ]
