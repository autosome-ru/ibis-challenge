from attr import define, field

from typing import Union, Optional
from tagger import UniqueTagger
from rwlock import RWLock

from sequence import Sequence

@define
class SeqDB:
    seq2tag: dict[str, str] = field(factory=dict)
    tag2seq: dict[str, str] = field(factory=dict)
    tagger: UniqueTagger = field(factory=UniqueTagger.default)
    lock: RWLock = RWLock() 
    
    def add(self, seq: Union[Sequence, str]) -> str:
        if isinstance(seq, Sequence):
            seq = seq.seq
        tag = self.get_tag(seq)
        if tag is None:
            with self.lock.w_locked(): 
                tag = self.tagger.tag()
                self.seq2tag[seq] = tag
                self.tag2seq[tag] = seq
        return tag

    def get_tag(self, seq: Union[Sequence, str]) -> Optional[str]:
        if isinstance(seq, Sequence):
            seq = seq.seq
        with self.lock.r_locked():
            tag = self.seq2tag.get(seq)
        return tag

    def get_seq(self, tag: str) -> Sequence:
        with self.lock.r_locked():
            seq = self.tag2seq[tag]
        return Sequence(seq)