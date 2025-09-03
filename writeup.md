## Problem(unicode1):
(a): Blank
(b): __repr__(chr(0)) == '\x00', hexadecimal 0,The binary representation is different.样
(c): 'this is a test\x00string' and "this is a teststring"

## Problem (unicode2): Unicode Encodings 
(a): UTF-16 and UTF-32 output list are too long
(b): UTF-8 is a variable-length encoding, with each character represented by 1 to 4 bytes.
(c): Sorry i don't know. I guess [255,255] ,because [255,254] is BOM

