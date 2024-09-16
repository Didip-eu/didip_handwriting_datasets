# For prototyping many-to-one charsets, based on existing transcriptions.

#[ (0x60+i, 0x60+i-32, chr(0x60+i), chr(0x60+i-32)) for i in range(1,27) ]
#[ (Oxe0+i, 0xe0+i-32, chr(0xe0+i), chr(0xe0+i-32)) for i in range(1,27) ]
#[ (i, chr(0x100+i)) for i in range(1,126) ]

As='aAáÁâÂãÃäÄåÅæÆāĂăĄą'
Bs='bB'
Cs='cCçÇĆćĈĉĊċČč'
Ds='dDðÐĎďĐđ'
Es='eEèÈéÉêÊëËĒēĔĕĖėĘęĚě'
Fs='fF'
Gs='gGĜĝĞğĠġĢģ'
Hs='hHĤĥĦħ'
Is='iIìÌíÍîÎïÏĨĩĪīĬĭĮįİıĲĳ'
Js='jJĴĵ'
Ks='kKĶķĸ'
Ls='lLĹĺĻļĽľĿŀŁł'
Ms='mM'
Ns='nNñÑŃńŅņŇňŉŊŋ'
Os='oOòÒóÓôÔõÕöÖŌōŎŏŐőŒœ'
Ps='pP'
Qs='qQ'
Rs='rRŔŕŖŗŘř'
Ss='sSŚśŜŝŞşŠš'
Ts='tTŢţŤťŦŧ'
Us='uUùÙúÚŨũŪūŬŭŮůŰűŲų'
Vs='vV'
Ws='wWŴŵ'
Xs='xX'
Ys='yYŶŷŸ'
Zs='zZŹźŻżŽ'

# { item[0]: item for item in [  eval(chr(char_range)+'s') for char_range in range(ord('A'), ord('Z')) ] }

