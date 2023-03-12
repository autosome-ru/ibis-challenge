from Bio.Seq import Seq

def gc(seq: Seq):
    cnt = 0
    for n in seq:
        match n:
            case 'G' | 'C' | 'g' | 'c':
                cnt += 1
            case 'N'| 'n':
                cnt += 0.5
    return cnt / len(seq)