import sqlite3
import time 

from Bio.Seq import Seq
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .tagger import UniqueTagger
from ..seq.seqentry import SeqEntry

@dataclass
class TagDatabase:
    db_path: Path
    wait_time: float
    tagger: UniqueTagger
    
    TAG_TABLE_NAME: ClassVar[str] = "__TAG__"
    
    @staticmethod
    def _table_exists(cur: sqlite3.Cursor, 
                      table_name: str) -> bool:
        args = (table_name, )
        ans = cur.execute(f"""SELECT name FROM sqlite_master WHERE type='table' AND name=?""", args).fetchall()
        return len(ans) > 0

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _create_table(self) -> None:       
        with self.get_connection() as con: 
            cur = con.cursor()
            if not self._table_exists(cur, 
                                      table_name=self.TAG_TABLE_NAME):
                try:
                    cur.execute(f"CREATE TABLE {self.TAG_TABLE_NAME} (seq TEXT NOT NULL UNIQUE, tag TEXT NOT NULL UNIQUE)")
                except sqlite3.OperationalError: # database has been created by other process
                    pass 

    @classmethod
    def make(cls, 
             db_path: Path | str, 
             tagger: UniqueTagger,
             wait_time: float = 0.1):
        self = cls(db_path=Path(db_path),
                   tagger=tagger,
                   wait_time=wait_time)
        self._create_table()
        return self
        
    @staticmethod
    def retrieve_db(cur: sqlite3.Cursor, 
                    table_name: str) -> dict[str, str]:
        query = f"SELECT * FROM {table_name}"
        ans = cur.execute(query).fetchall()
        answer = {k: v for k, v in ans}
        return answer
    
    @staticmethod
    def update_db(cur: sqlite3.Cursor, 
                  table_name: str,
                  dt: dict[str, str]):
        query = f"INSERT INTO {table_name} VALUES(?, ?)"
        cur.executemany(query, dt.items())
        
    @staticmethod
    def seq2str(s: str | Seq) -> str:
        return str(s).upper()
    
    @classmethod
    def seqs2strs(cls, 
                  seqs: list[str] | list[Seq] | list[str | Seq]) -> list[str]:
        mod_seqs = []
        for s in seqs:
            mod_seqs.append(cls.seq2str(s))
        return mod_seqs
                
    def taggify(self, 
            seqs: list[str] | list[Seq] | list[str | Seq] ) -> dict[str, str]:
        seqs = self.seqs2strs(seqs)
    
        wait_for_unlock = False
        existing = {}
        rest = {}
        while True:
            try:
                with self.get_connection() as con:
                    cur = con.cursor()
                    if not wait_for_unlock:
                        existing = self.retrieve_db(cur=cur, 
                                                    table_name=self.TAG_TABLE_NAME)
                        self.tagger.update_used(existing.values())
                        rest = {}
                        for s in seqs:
                            if s in existing:
                                continue
                            rest[s] = self.tagger.tag()
                    self.update_db(cur, self.TAG_TABLE_NAME, rest)  
            except sqlite3.IntegrityError as exc:
                print(f"Waiting: database lock exception ({exc}), regenerating keys")
                wait_for_unlock=False
            except sqlite3.OperationalError as exc:
                print(f"Waiting: database unique exception ({exc}), waiting for unlock")
                wait_for_unlock=True
                time.sleep(self.wait_time)
            else:
                break
                
        existing.update(rest)
        return {s: existing[s] for s in seqs}
    
    def taggify_entries(self, entries: list[SeqEntry]):
        entry_mapping = {self.seq2str(e.sequence) : e for e in entries}
        seq_tags = self.taggify(list(entry_mapping.keys()))
        for s, t in seq_tags.items():
            e = entry_mapping[s]
            e.tag = t
        return entries
    
    