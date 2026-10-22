"""Disk-backed full-text index for the complete project corpus, including table rows."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from collections.abc import Sequence
from datetime import datetime, timezone

from src.rag.knowledge_base import DocumentChunk
from src.rag.project_documents import ProjectReader, discover

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 2


class ChunkView(Sequence):
    """Compatibility view without loading every table row into RAM."""
    def __init__(self, index):
        self.index = index

    def __len__(self):
        with self.index.connect() as db:
            return db.execute('SELECT COALESCE(SUM(chunks),0) FROM files').fetchone()[0]

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            with self.index.connect() as db:
                rows = db.execute('SELECT payload FROM chunks ORDER BY rowid LIMIT ? OFFSET ?',
                                  (stop-start,start)).fetchall()
            return [DocumentChunk.from_dict(json.loads(r[0])) for r in rows][::step]
        if item < 0: item += len(self)
        with self.index.connect() as db:
            row = db.execute('SELECT payload FROM chunks ORDER BY rowid LIMIT 1 OFFSET ?', (item,)).fetchone()
        if row is None: raise IndexError(item)
        return DocumentChunk.from_dict(json.loads(row[0]))

    def __iter__(self):
        with self.index.connect() as db:
            for row in db.execute('SELECT payload FROM chunks ORDER BY rowid'):
                yield DocumentChunk.from_dict(json.loads(row[0]))


class ProjectIndexer:
    def __init__(self, config):
        self.config = config
        self.index_dir = config.knowledge_base.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / 'project.sqlite3'
        self.meta_path = self.index_dir / 'project_index_meta.json'
        self.inventory_path = self.index_dir / 'coverage.json'
        self.chunks_path = self.db_path
        self._lock = threading.RLock()
        self.file_hashes = {}
        self.knowledge_version = config.knowledge_base.version
        self.coverage = {}
        with self.connect() as db:
            # Readers keep serving the committed corpus while an update is built.
            if db.execute('PRAGMA journal_mode').fetchone()[0] != 'wal':
                db.execute('PRAGMA journal_mode=WAL')
            existing = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'chunks','search','files','settings'} <= existing:
                db.executescript('''
                CREATE TABLE IF NOT EXISTS chunks(id TEXT UNIQUE, source TEXT, payload TEXT);
                CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
                CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(title, body, tokenize='unicode61 remove_diacritics 2');
                CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, hash TEXT, status TEXT, chunks INTEGER, issues TEXT);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
                ''')
            self.file_hashes = dict(db.execute('SELECT path, hash FROM files'))
            row = db.execute("SELECT value FROM settings WHERE key='knowledge_version'").fetchone()
            if row: self.knowledge_version = row[0]
        if self.inventory_path.exists():
            try: self.coverage = json.loads(self.inventory_path.read_text(encoding='utf-8'))
            except (OSError,ValueError): pass
        self.chunks = ChunkView(self)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=60)
        db.execute('PRAGMA busy_timeout=60000')
        try:
            with db:
                yield db
        finally:
            db.close()

    def compute_file_hash(self, relative):
        path = self.config.project_root / relative
        digest = hashlib.sha256()
        # Model weight bytes do not need to be read to index architecture metadata.
        if path.suffix.lower() in {'.keras','.npy','.npz'}:
            stat = path.stat()
            digest.update(f'{stat.st_size}:{stat.st_mtime_ns}'.encode())
        else:
            with path.open('rb') as stream:
                for data in iter(lambda: stream.read(1024*1024), b''): digest.update(data)
            sidecar = path.with_suffix('.metadata.json')
            if path.suffix.lower()=='.pdf' and sidecar.is_file():
                from src.rag.project_documents import permitted
                if permitted(self.config,sidecar.relative_to(self.config.project_root).as_posix()):
                    with sidecar.open('rb') as stream:
                        digest.update(stream.read(1024*1024+1))
        return digest.hexdigest()

    def fingerprint(self):
        data = {'schema':SCHEMA_VERSION, 'version':self.config.knowledge_base.version,
                'chunk_chars':self.config.chunking.markdown_max_tokens*5,
                'exclusions':self.config.knowledge_base.exclude_patterns,
                'max_file_bytes':self.config.knowledge_base.max_file_bytes}
        return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

    @staticmethod
    def remove_source(db, source):
        ids = db.execute('SELECT rowid FROM chunks WHERE source=?',(source,)).fetchall()
        db.executemany('DELETE FROM search WHERE rowid=?', ids)
        db.execute('DELETE FROM chunks WHERE source=?',(source,))
        db.execute('DELETE FROM files WHERE path=?',(source,))

    def update_index(self, force=False):
        with self._lock:
            logger.info('Discovering project sources...')
            paths, coverage = discover(self.config)
            logger.info('Found %s readable sources; hashing changes...',len(paths))
            hashes = {}
            for path in paths:
                try: hashes[path] = self.compute_file_hash(path)
                except OSError as exc:
                    coverage.setdefault('read_errors', []).append({'source_path':path,'error':type(exc).__name__})
            with self.connect() as db:
                old = dict(db.execute('SELECT path, hash FROM files'))
                setting = db.execute("SELECT value FROM settings WHERE key='fingerprint'").fetchone()
                force = force or not setting or setting[0] != self.fingerprint()
                retry = {row[0] for row in db.execute("SELECT path FROM files WHERE status IN ('error','partial','empty')")}
                removed = sorted(set(old)-set(hashes))
                changed = [p for p in paths if p in hashes and (force or p in retry or old.get(p)!=hashes[p])]
                for path in removed: self.remove_source(db,path)
                for i,path in enumerate(changed,1):
                    self.remove_source(db,path)
                    reader = ProjectReader(self.config)
                    count = 0
                    status = 'indexed'
                    # Failed parsers must not leave half a CSV indexed as complete.
                    db.execute('SAVEPOINT ingest_file')
                    try:
                        for chunk in reader.read(path):
                            payload = json.dumps(chunk.to_dict(),ensure_ascii=False,separators=(',',':'))
                            rowid = db.execute('INSERT INTO chunks(id,source,payload) VALUES (?,?,?)',
                                               (chunk.chunk_id,path,payload)).lastrowid
                            db.execute('INSERT INTO search(rowid,title,body) VALUES (?,?,?)',
                                       (rowid,chunk.title,chunk.content))
                            count += 1
                        status = 'partial' if reader.issues else ('indexed' if count else 'empty')
                    except Exception as exc:
                        db.execute('ROLLBACK TO ingest_file')
                        count = 0; status = 'error'
                        reader.issues.append({'source_path':path,'status':'error','error':type(exc).__name__})
                        logger.warning('Cannot index %s (%s)',path,type(exc).__name__)
                    finally:
                        db.execute('RELEASE ingest_file')
                    db.execute('INSERT INTO files VALUES (?,?,?,?,?)',
                               (path,hashes[path],status,count,json.dumps(reader.issues,ensure_ascii=False)))
                    if i % 25 == 0: logger.info('Indexed %s/%s files',i,len(changed))
                now = datetime.now(timezone.utc).isoformat()
                if changed or removed:
                    version_hash = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
                    self.knowledge_version = self.config.knowledge_base.version + '-' + version_hash
                db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)',('fingerprint',self.fingerprint()))
                db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)',('knowledge_version',self.knowledge_version))
                statuses = [dict(zip(['source_path','status','chunks','issues'],r))
                            for r in db.execute('SELECT path,status,chunks,issues FROM files ORDER BY path')]
                for item in statuses: item['issues'] = json.loads(item['issues'])
                coverage.update({'indexed_at_utc':now,'knowledge_version':self.knowledge_version,
                                 'files':statuses,'indexed_files':sum(x['chunks']>0 for x in statuses),
                                 'complete_files':sum(x['status']=='indexed' for x in statuses),
                                 'paper_files':sum(x['source_path'].lower().endswith('.pdf') and x['chunks']>0 for x in statuses),
                                 'issues_count':sum(len(x['issues']) for x in statuses)})
                chunk_count = sum(item['chunks'] for item in statuses)
                coverage['chunks_count'] = chunk_count
            self.file_hashes = hashes
            self.coverage = coverage
            self.inventory_path.write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
            result = {'status':'rebuilt' if force else ('updated' if changed or removed else 'up_to_date'),
                      'files_count':coverage['indexed_files'],'chunks_count':chunk_count,
                      'knowledge_version':self.knowledge_version,
                      'added':[p for p in changed if p not in old],
                      'modified':[p for p in changed if p in old], 'removed':removed,
                      'paper_files':coverage['paper_files'],'issues_count':coverage['issues_count']}
            self.meta_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            report_dir = self.config.project_root / 'outputs' / 'rag_index'
            if report_dir.resolve() != self.index_dir.resolve():
                report_dir.mkdir(parents=True,exist_ok=True)
                (report_dir/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
                (report_dir/'project_index_meta.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            return result

    def rebuild(self):
        return self.update_index(force=True)

    def reload(self):
        with self.connect() as db:
            self.file_hashes = dict(db.execute('SELECT path,hash FROM files'))
            row = db.execute("SELECT value FROM settings WHERE key='knowledge_version'").fetchone()
            if row: self.knowledge_version = row[0]
        if self.inventory_path.exists():
            self.coverage = json.loads(self.inventory_path.read_text(encoding='utf-8'))

    def source_chunks(self, source, limit=3):
        with self.connect() as db:
            return [DocumentChunk.from_dict(json.loads(r[0])) for r in db.execute(
                'SELECT payload FROM chunks WHERE source=? ORDER BY rowid LIMIT ?', (source,limit))]

    def get_chunk(self, chunk_id):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM chunks WHERE id=?',(chunk_id,)).fetchone()
        return DocumentChunk.from_dict(json.loads(row[0])) if row else None

    def search(self, query, top_k=5, *, paper_only=False, source_paths=None):
        from src.rag.indexer import tokenize
        tokens = list(dict.fromkeys(tokenize(query)))[:48]
        if not tokens: return []
        # Small, explicitly scoped corpora do not need to rank every CSV row in FTS.
        if paper_only or source_paths:
            from src.rag.indexer import BM25TFIDFIndex
            paths = source_paths or [f['source_path'] for f in self.coverage.get('files',[])
                                    if f['source_path'].lower().endswith('.pdf') or
                                    f['source_path'].endswith('.notes.md') or f['source_path'].startswith('papers/')]
            if not paths:
                return []
            with self.connect() as db:
                rows = db.execute('SELECT payload FROM chunks WHERE source IN ('+','.join('?' for _ in paths)+')',paths).fetchall()
            chunks = [DocumentChunk.from_dict(json.loads(row[0])) for row in rows]
            if paper_only:
                chunks = [c for c in chunks if c.doc_type in ('paper','paper_note')]
            if not chunks:
                return []
            index = BM25TFIDFIndex()
            index.fit([c.title+'\n'+c.content for c in chunks])
            scores = index.score(query)
            return sorted([(c,float(s)) for c,s in zip(chunks,scores) if s>0],key=lambda pair:pair[1],reverse=True)[:top_k]
        expression = ' OR '.join('"'+t.replace('"','""')+'"' for t in tokens)
        with self.connect() as db:
            rows = db.execute('''SELECT c.payload, ranked.score FROM
                                 (SELECT rowid, rank AS score FROM search WHERE search MATCH ?
                                  AND rank MATCH 'bm25(3.0,1.0)' ORDER BY rank LIMIT ?) AS ranked
                                 JOIN chunks c ON c.rowid=ranked.rowid ORDER BY ranked.score''',
                              (expression,top_k)).fetchall()
        return [(DocumentChunk.from_dict(json.loads(payload)),max(0.0,-float(score))) for payload,score in rows]
