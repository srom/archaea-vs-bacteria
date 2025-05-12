"""
SOURCE: http://isoelectric.org/

Included explicitly because the PyPi package is broken.
"""
__author__ = "Lukasz Pawel Kozlowski"
__email__  = "lukaszkozlowski.lpk@gmail.com"
__copyrights__ = "Lukasz Pawel Kozlowski"
__webserver__ = "http://isoelectric.org"
__license__ = "http://isoelectric.org/license.txt"

# to ad new pKa sets just add new dictionary       
scales = {
"EMBOSS":     {'Cterm': 3.6, 'pKAsp': 3.9,  'pKGlu': 4.1, 'pKCys': 8.5, 'pKTyr': 10.1, 'pk_his': 6.5, 'Nterm': 8.6, 'pKLys': 10.8, 'pKArg': 12.5},
"DTASelect":  {'Cterm': 3.1, 'pKAsp': 4.4,  'pKGlu': 4.4, 'pKCys': 8.5, 'pKTyr': 10.0, 'pk_his': 6.5, 'Nterm': 8.0, 'pKLys': 10.0, 'pKArg': 12.0},
"Solomon":    {'Cterm': 2.4, 'pKAsp': 3.9,  'pKGlu': 4.3, 'pKCys': 8.3, 'pKTyr': 10.1, 'pk_his': 6.0, 'Nterm': 9.6, 'pKLys': 10.5, 'pKArg': 12.5}, 
"Sillero":    {'Cterm': 3.2, 'pKAsp': 4.0,  'pKGlu': 4.5, 'pKCys': 9.0, 'pKTyr': 10.0, 'pk_his': 6.4, 'Nterm': 8.2, 'pKLys': 10.4, 'pKArg': 12.0},
"Rodwell":    {'Cterm': 3.1, 'pKAsp': 3.68, 'pKGlu': 4.25,'pKCys': 8.33,'pKTyr': 10.07,'pk_his': 6.0, 'Nterm': 8.0, 'pKLys': 11.5, 'pKArg': 11.5},
"Patrickios": {'Cterm': 4.2, 'pKAsp': 4.2,  'pKGlu': 4.2, 'pKCys': 0.0, 'pKTyr':  0.0, 'pk_his': 0.0, 'Nterm': 11.2,'pKLys': 11.2, 'pKArg': 11.2},
"Wikipedia":  {'Cterm': 3.65,'pKAsp': 3.9,  'pKGlu': 4.07,'pKCys': 8.18,'pKTyr': 10.46,'pk_his': 6.04,'Nterm': 8.2, 'pKLys': 10.54,'pKArg': 12.48},
"Grimsley":   {'Cterm': 3.3, 'pKAsp': 3.5,  'pKGlu': 4.2, 'pKCys': 6.8, 'pKTyr': 10.3, 'pk_his': 6.6, 'Nterm': 7.7, 'pKLys': 10.5, 'pKArg': 12.04},
'Lehninger':  {'Cterm': 2.34,'pKAsp': 3.86, 'pKGlu': 4.25,'pKCys': 8.33,'pKTyr': 10.0, 'pk_his': 6.0, 'Nterm': 9.69,'pKLys': 10.5, 'pKArg': 12.4},
'Bjellqvist': {'Cterm': 3.55,'pKAsp': 4.05, 'pKGlu': 4.45,'pKCys': 9.0, 'pKTyr': 10.0, 'pk_his': 5.98,'Nterm': 7.5, 'pKLys': 10.0, 'pKArg': 12.0},   
'IPC_peptide':{'Cterm': 2.383, 'pKAsp': 3.887, 'pKGlu': 4.317, 'pKCys': 8.297, 'pKTyr': 10.071, 'pk_his': 6.018, 'Nterm': 9.564, 'pKLys': 10.517, 'pKArg': 12.503},    # IPC peptide
'IPC_protein':{'Cterm': 2.869, 'pKAsp': 3.872, 'pKGlu': 4.412, 'pKCys': 7.555, 'pKTyr': 10.85,  'pk_his': 5.637, 'Nterm': 9.094, 'pKLys': 9.052,  'pKArg': 11.84},     # IPC protein 
'Toseland':   {'Cterm': 3.19,'pKAsp': 3.6,  'pKGlu': 4.29,'pKCys': 6.87,'pKTyr': 9.61, 'pk_his': 6.33,'Nterm': 8.71, 'pKLys': 10.45, 'pKArg':  12},
'Thurlkill':  {'Cterm': 3.67,'pKAsp': 3.67, 'pKGlu': 4.25,'pKCys': 8.55,'pKTyr': 9.84, 'pk_his': 6.54,'Nterm': 8.0, 'pKLys': 10.4, 'pKArg': 12.0},
'Nozaki':     {'Cterm': 3.8, 'pKAsp': 4.0,  'pKGlu': 4.4, 'pKCys': 9.5, 'pKTyr': 9.6,  'pk_his': 6.3, 'Nterm': 7.5, 'pKLys': 10.4, 'pKArg': 12},   
'Dawson':     {'Cterm': 3.2, 'pKAsp': 3.9,  'pKGlu': 4.3, 'pKCys': 8.3, 'pKTyr': 10.1, 'pk_his': 6.0, 'Nterm': 8.2, 'pKLys': 10.5, 'pKArg':  12},   
          }

aaDict = {'Asp':'D', 'Glu':'E', 'Cys':'C', 'Tyr':'Y', 'His':'H', 
          'Lys':'K', 'Arg':'R', 'Met':'M', 'Phe':'F', 'Leu':'L', 
          'Val':'V', 'Ala':'A', 'Gly':'G', 'Gln':'Q', 'Asn':'N',
          'Ile':'I', 'Trp':'W', 'Ser':'S', 'Thr':'T', 'Sec':'U',
          'Pro':'P', 'Xaa':'X', 'Sec':'U', 'Pyl':'O', 'Asx':'B',
          'Xle':'J', }

acidic = ['D', 'E', 'C', 'Y']
basic = ['K', 'R', 'H']

pKcterminal = {'D': 4.55, 'E': 4.75} 
pKnterminal = {'A': 7.59, 'M': 7.0, 'S': 6.93, 'P': 8.36, 'T': 6.82, 'V': 7.44, 'E': 7.7} 

def predict_isoelectric_point(seq, scale='IPC_protein'):
    """calculate isoelectric point using 9 pKa set model"""
    pKCterm = scales[scale]['Cterm']
    pKAsp = scales[scale]['pKAsp']
    pKGlu = scales[scale]['pKGlu']
    pKCys = scales[scale]['pKCys']
    pKTyr = scales[scale]['pKTyr']
    pKHis = scales[scale]['pk_his']
    pKNterm = scales[scale]['Nterm']
    pKLys = scales[scale]['pKLys'] 
    pKArg = scales[scale]['pKArg']
    pH = 6.51             #starting po pI = 6.5 - theoretically it should be 7, but average protein pI is 6.5 so we increase the probability of finding the solution
    pHprev = 0.0         
    pHnext = 14.0        
    E = 0.01             #epsilon means precision [pI = pH +- E]
    temp = 0.01
    nterm=seq[0]
    if scale=='Bjellqvist':
        if nterm in pKnterminal.keys():
            pKNterm = pKnterminal[nterm]
    
    cterm=seq[-1]
    if scale=='Bjellqvist':
        if cterm in pKcterminal.keys():
            pKCterm = pKcterminal[cterm] 
            
    while 1:             #the infinite loop
        QN1=-1.0/(1.0+pow(10,(pKCterm-pH)))                                        
        QN2=-seq.count('D')/(1.0+pow(10,(pKAsp-pH)))           
        QN3=-seq.count('E')/(1.0+pow(10,(pKGlu-pH)))           
        QN4=-seq.count('C')/(1.0+pow(10,(pKCys-pH)))           
        QN5=-seq.count('Y')/(1.0+pow(10,(pKTyr-pH)))        
        QP1=seq.count('H')/(1.0+pow(10,(pH-pKHis)))            
        QP2=1.0/(1.0+pow(10,(pH-pKNterm)))                
        QP3=seq.count('K')/(1.0+pow(10,(pH-pKLys)))           
        QP4=seq.count('R')/(1.0+pow(10,(pH-pKArg)))            
        NQ=QN1+QN2+QN3+QN4+QN5+QP1+QP2+QP3+QP4
        #print NQ
        #%%%%%%%%%%%%%%%%%%%%%%%%%   BISECTION   %%%%%%%%%%%%%%%%%%%%%%%%
        if NQ<0.0:              #we are out of range, thus the new pH value must be smaller                     
            temp = pH
            pH = pH-((pH-pHprev)/2.0)
            pHnext = temp
            #print "pH: ", pH, ", \tpHnext: ",pHnext
        else:
            temp = pH
            pH = pH + ((pHnext-pH)/2.0)
            pHprev = temp
            #print "pH: ", pH, ",\tpHprev: ", pHprev

        if (pH-pHprev<E) and (pHnext-pH<E): #terminal condition, finding pI with given precision
            return pH

def calculate_molecular_weight(seq):
    """molecular weight"""
    massDict = {'D':115.0886, 'E': 129.1155, 'C': 103.1388, 'Y':163.1760, 'H':137.1411, 
           'K':128.1741, 'R': 156.1875,  'M': 131.1926, 'F':147.1766, 'L':113.1594,
           'V':99.1326,  'A':  71.0788,  'G':  57.0519, 'Q':128.1307, 'N':114.1038, 
           'I':113.1594, 'W': 186.2132,  'S':  87.0782, 'P': 97.1167, 'T':101.1051, 
           'U':141.05,   'h2o':18.01524, 'X':        0, 'Z':128.6231, 'O':255.31, 
           'B':114.5962, 'J': 113.1594,}
    molecular_weight = massDict['h2o']
    for aa in seq:
            molecular_weight+=massDict[aa]       
    return molecular_weight
