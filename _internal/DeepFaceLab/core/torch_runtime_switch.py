"""Transactional replacement of the two pinned Torch CUDA packages only.

The manager is standard-library-only. Native Torch is loaded in short-lived
probe children, so the process moving package directories never locks a DLL.
"""
from contextlib import contextmanager
import configparser
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zipfile

from .gpu_runtime import architecture_supported, installed_runtime, load_manifest, runtime_sbom

REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
NAMES = ('torch', 'torchvision')
PACKAGE_ROOTS = {'torch', 'torchgen', 'functorch', 'torchvision', 'torchvision.libs'}
SCRIPT_NAMES = {'torchrun', 'torchfrtrace', 'convert-caffe2-to-onnx', 'convert-onnx-to-caffe2'}
COMMITTED = {'active', 'kept', 'deleted', 'rolled-back'}
INCOMPLETE = {'switching', 'verifying-active', 'rolling-back', 'recovery-required'}
TOKEN = re.compile(r'^[0-9a-f]{32}$')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _existing_lock_busy(path):
    """Probe a pre-existing manager lock without creating or changing it."""
    if not path.exists():
        return False
    try:stream=path.open('r+b')
    except PermissionError:return True  # .NET source installer holds FileShare.None.
    with stream:
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        stream.seek(0)
        if os.name == 'nt':
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def assert_runtime_available(root, maintenance_transaction=None):
    """Call before importing Torch in normal entry points; never import Torch here."""
    root = Path(os.path.abspath(root))
    source=root/'.launcher-install/source'
    for path in [source,source/'source-install.lock',source/'source-install-owner.json',source/'python-repair.json']:
        for ancestor in [path,*path.parents]:
            try:info=ancestor.lstat()
            except FileNotFoundError:info=None
            if info and (getattr(info,'st_file_attributes',0)&REPARSE or stat.S_ISLNK(info.st_mode)):
                raise RuntimeError('运行环境安装路径异常，请先修复后再启动。')
            if ancestor==root:break
    repair_path=source/'python-repair.json'
    repair=None;private=False
    if repair_path.exists():
        try:
            repair=json.loads(repair_path.read_text(encoding='utf-8-sig'));token=repair.get('transactionId','')
            if (repair.get('schemaVersion')!=1 or not TOKEN.fullmatch(token)
                    or repair.get('phase') not in {'preparing','moving-old','creating','installing','verifying','committed','restoring','rolled-back'}
                    or repair.get('backup')!='.launcher-install/source/venv-backup-'+token
                    or repair.get('session')!='.launcher-install/source/python-session-'+token
                    or type(repair.get('hadVenv')) is not bool or repair.get('phase')=='committed' and repair.get('validated') is not True):
                raise ValueError('Invalid Python repair journal')
            if maintenance_transaction is not None:
                owner=json.loads((source/'source-install-owner.json').read_text(encoding='utf-8-sig'))
                private=(maintenance_transaction==token and TOKEN.fullmatch(repair.get('lockToken','')) and owner.get('schemaVersion')==1
                         and owner.get('token')==repair.get('lockToken') and owner.get('pid')==repair.get('lockOwnerPid')
                         and _pid_alive(owner.get('pid')) and _existing_lock_busy(source/'source-install.lock'))
                if not private:raise ValueError('Invalid private maintenance validation token')
        except (OSError,ValueError,TypeError) as error:raise RuntimeError('Python 环境修复记录异常，请先恢复。') from error
    elif maintenance_transaction is not None:
        raise RuntimeError('私有环境验证需要匹配的安装事务。')
    if not private and (_existing_lock_busy(source/'source-install.lock') or repair and repair['phase'] not in {'committed','rolled-back'}):
        raise RuntimeError('Python 运行环境正在安装或等待恢复，请先在启动器完成恢复。')
    directory = root / '.runtime/torch-variants'
    for path in [directory, directory/'operation.lock', directory/'switch.json']:
        for ancestor in [path, *path.parents]:
            try:info=ancestor.lstat()
            except FileNotFoundError:info=None
            if info is not None:
                if info.st_file_attributes & REPARSE or stat.S_ISLNK(info.st_mode):
                    raise RuntimeError('运行环境管理路径异常，请先修复后再启动。')
            if ancestor == root:
                break
    try:
        if _existing_lock_busy(directory/'operation.lock'):
            raise RuntimeError('运行环境正在下载、切换或清理，请等待完成后再启动任务。')
        state_path = directory/'switch.json'
        if state_path.exists():
            state = RuntimeSwitch(root)._state()
            if state.get('phase') not in COMMITTED | {'failed', 'deleting'}:
                raise RuntimeError('运行环境操作尚未完成，请先在启动器恢复原环境。')
    except (OSError, ValueError) as error:
        raise RuntimeError('无法确认运行环境状态，请先在启动器检查或恢复。') from error


def _pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        process = kernel.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(process, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(process)
    return Path('/proc', str(pid)).exists()


class RuntimeSwitch:
    def __init__(self, root, progress=None):
        self.root = Path(os.path.abspath(root))
        self._checked(self.root)
        if not (self.root / '_internal/DeepFaceLab/me.py').is_file():
            raise ValueError('Not a complete DFL-PT-WEBUI source project')
        self.manifest = load_manifest(self.root)
        self.active = self.root / '.venv'
        self.managed = self.root / '.runtime/torch-variants'
        self.journal_path = self.managed / 'switch.json'
        self.progress = progress or (lambda value: None)

    def _checked(self, path):
        path = Path(os.path.abspath(path))
        if path != self.root and not path.is_relative_to(self.root):
            raise ValueError('Runtime path escapes the project')
        for ancestor in [path, *path.parents]:
            try:info=ancestor.lstat()
            except FileNotFoundError:info=None
            if info is not None:
                if info.st_file_attributes & REPARSE or stat.S_ISLNK(info.st_mode):
                    raise ValueError('Linked runtime paths are not supported: ' + str(ancestor))
            if ancestor == self.root:
                break
        return path

    def _mkdir(self, path):
        path = self._checked(path)
        path.mkdir(parents=True, exist_ok=True)
        return self._checked(path)

    def _read(self, path):
        return json.loads(self._checked(path).read_text(encoding='utf-8'))

    def _write(self, path, value):
        path = self._checked(path)
        self._mkdir(path.parent)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(value, stream, ensure_ascii=False, separators=(',', ':'))
                stream.flush()
                os.fsync(stream.fileno())
            self._checked(path)
            os.replace(temporary, path)
        finally:
            # Only this invocation's exclusive temporary file is ours to remove.
            if temporary.exists():
                try:self._checked(temporary).unlink()
                except OSError:pass

    def _state(self):
        if not self.journal_path.exists():
            return None
        state = self._read(self.journal_path)
        if state.get('schemaVersion') != 1 or not TOKEN.fullmatch(state.get('transactionId', '')):
            raise ValueError('Invalid runtime transaction record; retained files were not changed')
        if (state.get('sourceRuntime') not in self.manifest['runtimes'] or state.get('targetRuntime') not in self.manifest['runtimes']
                or state['sourceRuntime'] == state['targetRuntime'] or state.get('phase') not in COMMITTED | INCOMPLETE | {'downloading','staging','probing','failed','deleting'}):
            raise ValueError('Invalid runtime transaction phase or variant')
        for key, parent in [('stage', 'staged'), ('backup', 'retained'), ('work', 'transactions')]:
            expected = self.managed / parent / state['transactionId']
            # Reusing a retained version is deliberately not inferred from UI paths.
            if self._state_path(state[key]) != expected:
                raise ValueError('Runtime transaction path does not match its identifier')
        reuse=state.get('reuseTransactionId')
        if reuse is not None and (not TOKEN.fullmatch(reuse) or reuse==state['transactionId']):
            raise ValueError('Invalid retained runtime source')
        expected_source=self.managed/'retained'/reuse if reuse else self._state_path(state['stage'])
        if 'newSource' in state and self._state_path(state['newSource'])!=expected_source:
            raise ValueError('Runtime package source does not match its retained identifier')
        for kind, runtime in [('old',state['sourceRuntime']),('new',state['targetRuntime'])]:
            entries = state.get(kind+'Entries', [])
            inventory = state.get(kind+'Inventory', [])
            self._validate_entries(entries, runtime)
            self._validate_inventory(inventory, entries)
        if not isinstance(state.get('steps'),list):
            raise ValueError('Invalid runtime transaction steps')
        seen = set()
        for step in state['steps']:
            key=(step.get('kind'),step.get('path'))
            if (step.get('kind') not in {'old','new'} or step.get('status') not in {'intent','done','rolled-back'}
                    or step.get('path') not in state.get(step['kind']+'Entries',[]) or key in seen):
                raise ValueError('Invalid runtime move record')
            seen.add(key)
        if state['phase'] in INCOMPLETE | {'active','kept','deleted','deleting','rolled-back'}:
            for kind,runtime in [('old',state['sourceRuntime']),('new',state['targetRuntime'])]:
                mandatory={'Lib/site-packages/'+name for name in NAMES} | {
                    'Lib/site-packages/'+name+'-'+self.manifest['runtimes'][runtime][name]+'.dist-info' for name in NAMES}
                if not mandatory<=set(state.get(kind+'Entries',[])) or not state.get(kind+'Inventory'):
                    raise ValueError('Incomplete runtime ownership plan')
            if state.get('retainedBytes')!=sum(row['size'] for row in state['oldInventory']):
                raise ValueError('Retained runtime size differs from its verified inventory')
        if state['phase'] in {'active','kept','deleted','deleting'}:
            expected={(kind,path) for kind in ['old','new'] for path in state[kind+'Entries']}
            if seen!=expected or any(step['status']!='done' for step in state['steps']) or not state.get('activeProbe',{}).get('passed'):
                raise ValueError('Runtime transaction lacks a completed activation proof')
        if state['phase'] in {'downloading','staging','probing','failed'} and state['steps']:
            raise ValueError('Preparation state contains unexpected package moves')
        if 'cacheWheelFiles' in state:
            files=state['cacheWheelFiles']
            known={runtime_sbom(self.root,state['targetRuntime'])['packages'][name]['wheel']['file'] for name in NAMES}
            if not isinstance(files,list) or len(files)!=len(set(files)) or any(name not in known for name in files):
                raise ValueError('Invalid cached wheel cleanup ownership')
        return state

    def _safe_relative(self, relative):
        if (not isinstance(relative,str) or not relative or ':' in relative or '\\' in relative or '\0' in relative
                or PurePosixPath(relative).is_absolute() or any(p in {'','.','..'} for p in relative.split('/'))):
            raise ValueError('Invalid relative runtime package path')
        return PurePosixPath(relative).parts

    def _validate_entries(self, entries, runtime):
        if not isinstance(entries,list) or len(entries)!=len(set(entries)):
            raise ValueError('Invalid owned runtime entries')
        allowed=PACKAGE_ROOTS | {f'{name}-{self.manifest["runtimes"][runtime][name]}.dist-info' for name in NAMES}
        for relative in entries:
            parts=self._safe_relative(relative)
            package=len(parts)==3 and parts[:2]==('Lib','site-packages') and parts[2] in allowed
            script=len(parts)==2 and parts[0]=='Scripts' and any(parts[1] in {name+'.exe',name+'-script.py',name} for name in SCRIPT_NAMES)
            if not(package or script):
                raise ValueError('Runtime entry is outside the two-package ownership boundary')

    def _validate_inventory(self, inventory, entries):
        if not isinstance(inventory,list):
            raise ValueError('Invalid owned runtime inventory')
        seen=set()
        for row in inventory:
            path=row.get('path');self._safe_relative(path)
            if (path in seen or not any(path==entry or path.startswith(entry+'/') for entry in entries)
                    or type(row.get('size')) is not int or row['size']<0 or type(row.get('mtime_ns')) is not int or row['mtime_ns']<0):
                raise ValueError('Runtime inventory escapes its owned entries')
            seen.add(path)

    def _save(self, state):
        state['updatedAt'] = _now()
        self._write(self.journal_path, state)

    def _relative(self, path):
        return self._checked(path).relative_to(self.root).as_posix()

    def _state_path(self, relative):
        value = PurePosixPath(relative)
        if value.is_absolute() or '..' in value.parts or ':' in relative or '\\' in relative:
            raise ValueError('Invalid managed runtime path')
        return self._checked(self.root / Path(*value.parts))

    @contextmanager
    def _lease(self):
        self._mkdir(self.managed)
        path = self._checked(self.managed / 'operation.lock')
        with path.open('a+b') as lock:
            if path.stat().st_size == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError('另一个运行环境操作正在进行，请等待完成。') from error
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _hardware(self):
        try:
            result = subprocess.run(['nvidia-smi.exe', '--query-gpu=name,compute_cap', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=8,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if result.returncode:
                return []
            hardware = []
            for fields in csv.reader(result.stdout.splitlines()):
                if len(fields) != 2 or not re.fullmatch(r'\d+\.\d+', fields[1].strip()):
                    return []
                hardware.append(dict(name=fields[0].strip(), capability=tuple(map(int, fields[1].strip().split('.')))))
            return hardware
        except (OSError, subprocess.TimeoutExpired):
            return []

    def _busy_reason(self):
        if os.name != 'nt':
            # This product's package switch is Windows-only; tests replace this probe.
            return '运行环境切换仅支持 Windows。'
        root = str(self.root).replace("'", "''")
        pattern = r'^(python|pythonw|node|ffmpeg|ffprobe)\.exe$'
        code = ("$ErrorActionPreference='Stop'; $p=@(Get-CimInstance Win32_Process | "
                "Where-Object { $_.ProcessId -ne " + str(os.getpid()) + " -and "
                "$_.Name -match '"+pattern+"' -and "
                "(($_.ExecutablePath -and $_.ExecutablePath.StartsWith('" + root + "\\',[StringComparison]::OrdinalIgnoreCase)) -or "
                "($_.CommandLine -and $_.CommandLine.IndexOf('"+root+"',[StringComparison]::OrdinalIgnoreCase) -ge 0)) }); "
                "if($p.Count){Write-Output 'busy'}")
        try:
            result = subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', code],
                capture_output=True, text=True, encoding='utf-8',errors='replace',timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode:
                return '无法确认项目任务是否已停止，请重新检测后再试。'
            return '请先保存并结束训练、WebUI 和项目终端，再切换运行环境。' if result.stdout.strip() else None
        except (OSError, subprocess.TimeoutExpired):
            return '无法确认项目任务是否已停止，请重新检测后再试。'

    def _assert_idle(self):
        reason = self._busy_reason()
        if reason:
            raise RuntimeError(reason)

    def _supports(self, runtime, capability):
        if capability[0] < 5:
            return False
        spec = self.manifest['runtimes'][runtime]
        return architecture_supported(capability, spec['architectures'])

    def _retained(self):
        directory = self._checked(self.managed / 'retained')
        entries={entry.name:self._checked(entry) for entry in directory.iterdir()} if directory.exists() else {}
        # The package directory may already be gone when a process dies before
        # committing deletion. Keep its confirmed tombstone visible for retry.
        transactions=self._checked(self.managed/'transactions')
        if transactions.exists():
            for work in transactions.iterdir():
                self._checked(work)
                if not TOKEN.fullmatch(work.name) or work.name in entries or not (work/'delete.json').exists():continue
                deletion=self._deletion_record(work.name)
                if not deletion['complete'] or deletion.get('cleanupWarning'):entries[work.name]=self._checked(directory/work.name)
        rows = []
        state=self._state()
        consumed=state.get('reuseTransactionId') if state and state['phase'] in {'active','kept','deleted','deleting'} else None
        for entry in entries.values():
            receipt = entry / 'receipt.json'
            if not TOKEN.fullmatch(entry.name):
                continue
            deletion=self._deletion_record(entry.name)
            if receipt.is_file():item=self._read(receipt)
            else:
                item=deletion
                if item is None:continue
            if deletion and deletion.get('complete') and entry.exists():raise ValueError('Unexpected retained directory after completed deletion')
            if (item.get('schemaVersion') != 1 or item.get('transactionId') != entry.name
                    or item.get('runtime') not in self.manifest['runtimes']):
                raise ValueError('Invalid retained runtime record')
            if item.get('deleted') or item.get('consumed') or entry.name==consumed:
                continue
            self._validate_entries(item.get('entries'),item['runtime'])
            self._validate_inventory(item.get('inventory'),item['entries'])
            if item.get('bytes')!=sum(row['size'] for row in item['inventory']):
                raise ValueError('Retained runtime size differs from its receipt')
            deleting=bool(item.get('deleteConfirmed') or deletion)
            rows.append(dict(transactionId=entry.name, runtime=item['runtime'],
                             bytes=0 if deletion and deletion.get('complete') else item['bytes'], kept=bool(item.get('kept')),deleting=deleting,
                             canReuse=not deleting and all((entry/path).exists() for path in item['entries'])))
        return rows

    def _matching_retained(self,runtime):
        rows=[row for row in self._retained() if row['runtime']==runtime and row['kept'] and row['canReuse']]
        return rows[-1] if rows else None

    def _new_source(self,state):
        return self._state_path(state.get('newSource',state['stage']))

    def status(self):
        state = self._state()
        pending = state and state['phase'] in INCOMPLETE
        current = state['sourceRuntime'] if pending else installed_runtime(self.root, self.manifest)
        hardware = self._hardware()
        result = dict(status='compatible', currentRuntime=current, targetRuntime=None,
            currentLabel=self._label(current), targetLabel=None, gpuNames=[h['name'] for h in hardware],
            message='', downloadBytes=0, retainedBytes=0, transactionId=None, retainedRuntime=None,
            cachedBytes=0,downloadRemainingBytes=0,
            cleanupWarning=None,canRetryCleanup=False,
            downloadRequired=True, canSwitch=False, canDelete=False, canRecover=False, error=None,
            retainedRuntimes=self._retained())
        stable=not state or state['phase'] in COMMITTED | {'failed','deleting'}
        can_delete=bool(stable and current in self.manifest['runtimes'] and hardware
                        and all(self._supports(current,h['capability']) for h in hardware))
        for item in result['retainedRuntimes']:
            eligible=(item['kept'] or item['deleting'] or state and item['transactionId']==state['transactionId']
                      and state['phase'] in {'active','kept','deleting'})
            item['canDelete']=bool(can_delete and eligible)
        if state:
            result.update(transactionId=state['transactionId'], retainedRuntime=state['sourceRuntime'],
                          retainedBytes=state.get('retainedBytes', 0),downloadRequired=not bool(state.get('reuseTransactionId')),
                          downloadBytes=state.get('downloadBytes',0),cachedBytes=state.get('cachedBytes',0),
                          downloadRemainingBytes=state.get('downloadRemainingBytes',0))
            if state.get('cleanupWarning'):
                result.update(cleanupWarning=state['cleanupWarning'],canRetryCleanup=state['phase'] in {'active','kept','deleted'})
        if pending:
            live = _pid_alive(state.get('ownerPid')) and _existing_lock_busy(self.managed/'operation.lock')
            result.update(status='switching' if live else 'error', canRecover=not live, error=state.get('error'),
                          message=state.get('message','正在切换运行环境。') if live else '上次运行环境切换未完成，请先恢复原环境。')
            return result
        if state and state['phase'] in {'downloading', 'staging', 'probing'}:
            if not _pid_alive(state.get('ownerPid')) or not _existing_lock_busy(self.managed/'operation.lock'):
                result.update(status='error', canRecover=True, message='上次下载或验证已中断，原环境保持不变。请先恢复后重试。')
                return result
            result.update(status='downloading', targetRuntime=state['targetRuntime'],
                          targetLabel=self._label(state['targetRuntime']), message=state.get('message', '正在准备匹配的运行环境。'))
            return result
        if state and state['phase'] in {'failed','rolled-back'} and state.get('cleanupWarning'):
            result.update(status='error',canRecover=True,error=state['cleanupWarning'],
                          message='上次准备留下的临时文件未清理，请重试恢复。原运行环境保持可用。')
            return result
        if state and state['phase'] in {'active', 'deleting'} and current == state['targetRuntime']:
            result.update(status='cleanup-choice', canDelete=can_delete,
                          targetRuntime=current, targetLabel=self._label(current),
                          message=('显卡信息再次变化，请先保留旧版并重新检测。' if not can_delete else
                                   '上次旧版清理未完成，可重试清理。' if state['phase']=='deleting' else
                                   '新运行环境已切换并验证通过，请选择保留或删除旧版。'))
            return result
        if not current:
            result.update(status='unavailable', message='请先完成项目运行环境安装。')
            return result
        if not hardware:
            result.update(status='unavailable', message='暂时无法读取显卡信息，当前运行环境保持不变。')
            return result
        candidates = [runtime for runtime in self.manifest['runtimes']
                      if all(self._supports(runtime, h['capability']) for h in hardware)]
        if not candidates:
            result.update(status='ambiguous', message='检测到需要不同运行环境的显卡，请选择要使用的显卡类型；当前环境保持不变。')
            return result
        if current in candidates:
            if state and state['phase'] == 'kept':
                result.update(status='retained',canDelete=can_delete, message='旧版运行环境已保留。')
            return result
        target = candidates[0]
        records = runtime_sbom(self.root, target)['packages']
        cached=sum(records[name]['wheel']['sizeBytes'] for name in NAMES if self._cached_wheel(target,records[name]['wheel']))
        total=sum(records[name]['wheel']['sizeBytes'] for name in NAMES)
        result.update(status='change-required', targetRuntime=target, targetLabel=self._label(target),
            message='检测到显卡环境变化，当前版本不适配这台电脑。可下载并切换到匹配的运行环境。',
            downloadRequired=cached<total,cachedBytes=cached,downloadRemainingBytes=total-cached,downloadBytes=total)
        if cached==total:result['message']='安装包已下载并校验，可准备并切换到匹配的运行环境。'
        if self._matching_retained(target):
            result.update(downloadRequired=False,downloadBytes=0,cachedBytes=0,downloadRemainingBytes=0,message='已保留匹配的运行环境，可验证后直接切换。')
        busy = self._busy_reason()
        result['canSwitch'] = busy is None
        if busy:
            result['message'] = busy
        if state and state['phase'] in {'failed', 'rolled-back'} and state.get('error'):
            result['error'] = state['error']
        return result

    def _label(self, runtime):
        return 'CUDA ' + self.manifest['runtimes'][runtime]['cuda'] if runtime else '未安装'

    def _emit(self, state, phase, message, **values):
        state.update(phase=phase, message=message, **values)
        self._save(state)
        self.progress(dict(status='switching' if phase in INCOMPLETE else 'downloading',
                           currentRuntime=state['sourceRuntime'], targetRuntime=state['targetRuntime'],
                           currentLabel=self._label(state['sourceRuntime']), targetLabel=self._label(state['targetRuntime']),
                           transactionId=state['transactionId'], message=message, canSwitch=False, canDelete=False,
                           downloadRequired=not bool(state.get('reuseTransactionId')) and state.get('downloadRemainingBytes',state.get('downloadBytes',0))>0,
                           cachedBytes=state.get('cachedBytes',0),downloadRemainingBytes=state.get('downloadRemainingBytes',0),
                           **{k: v for k, v in values.items() if k in {'downloadedBytes', 'downloadBytes'}}))

    def _digest(self, path):
        with self._checked(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def _cache_receipt(self,runtime,wheel):
        return self._checked(self.managed/'wheels'/runtime/(wheel['file']+'.verified.json'))

    def _cached_wheel(self,runtime,wheel):
        path=self._checked(self.managed/'wheels'/runtime/wheel['file'])
        receipt=self._cache_receipt(runtime,wheel)
        if not path.is_file() or not receipt.is_file():return False
        try:item=self._read(receipt)
        except (OSError,ValueError):return False
        info=path.stat()
        return (item.get('schemaVersion')==1 and item.get('runtime')==runtime and item.get('file')==wheel['file']
                and item.get('sha256')==wheel['sha256'] and item.get('sizeBytes')==wheel['sizeBytes']==info.st_size
                and item.get('mtimeNs')==info.st_mtime_ns)

    def _record_cached_wheel(self,runtime,wheel,path):
        info=self._checked(path).stat()
        self._write(self._cache_receipt(runtime,wheel),dict(schemaVersion=1,runtime=runtime,file=wheel['file'],
            sha256=wheel['sha256'],sizeBytes=info.st_size,mtimeNs=info.st_mtime_ns))

    def _download(self, state, name, item):
        wheel = item['wheel']
        directory = self._mkdir(self.managed / 'wheels' / state['targetRuntime'])
        target = self._checked(directory / wheel['file'])
        claimed=self._cached_wheel(state['targetRuntime'],wheel)
        if target.exists() and target.stat().st_size == wheel['sizeBytes'] and self._digest(target) == wheel['sha256']:
            self._record_cached_wheel(state['targetRuntime'],wheel,target)
            state.setdefault('downloadCacheHits',[]).append(name)
            if not claimed and 'cachedBytes' in state:
                state['cachedBytes']+=wheel['sizeBytes'];state['downloadRemainingBytes']=max(0,state['downloadRemainingBytes']-wheel['sizeBytes'])
            return target
        if claimed and 'cachedBytes' in state:
            state['cachedBytes']=max(0,state['cachedBytes']-wheel['sizeBytes']);state['downloadRemainingBytes']+=wheel['sizeBytes']
        url = wheel.get('url') or (self.manifest['runtimes'][state['targetRuntime']]['indexUrl'] + '/' + urllib.parse.quote(wheel['file']))
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname not in {'download.pytorch.org', 'download-r2.pytorch.org'}:
            raise ValueError('Only the pinned official PyTorch download is allowed')
        partial = target.with_name(target.name + '.' + state['transactionId'] + '.partial')
        self._checked(partial)
        if partial.exists():
            partial.unlink()
        received, last = 0, time.monotonic()
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as output:
            final = urllib.parse.urlsplit(response.url)
            if final.scheme != 'https' or final.hostname not in {'download.pytorch.org', 'download-r2.pytorch.org'}:
                raise ValueError('Unexpected PyTorch download redirect')
            for block in iter(lambda: response.read(4 * 1024**2), b''):
                received += len(block)
                if received > wheel['sizeBytes']:
                    raise ValueError('Downloaded wheel exceeds the pinned size')
                output.write(block)
                digest.update(block)
                if time.monotonic() - last > 1:
                    self._emit(state, 'downloading', '正在下载匹配的运行环境。', downloadedBytes=state.get('completedDownloadBytes', 0)+received,
                               downloadBytes=state['downloadBytes'])
                    last = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        if received != wheel['sizeBytes'] or digest.hexdigest() != wheel['sha256']:
            self._write(self._state_path(state['work'])/(name+'-download-failure.json'),
                dict(expectedBytes=wheel['sizeBytes'],actualBytes=received,expectedSha256=wheel['sha256'],actualSha256=digest.hexdigest()))
            raise ValueError('Official wheel size or SHA-256 verification failed: ' + name)
        os.replace(partial, target)
        self._record_cached_wheel(state['targetRuntime'],wheel,target)
        return target

    def _python(self):
        path = self._checked(self.active / 'Scripts/python.exe')
        if not path.is_file():
            raise ValueError('The existing project Python is missing')
        return path

    def _environment(self, work):
        temp = self._mkdir(work / 'tmp')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp),
                   NN_PROCESS_COUNT='0', CUDA_CACHE_PATH=str(temp/'cuda-cache'), TORCH_EXTENSIONS_DIR=str(temp/'torch-extensions'),
                   XDG_CACHE_HOME=str(temp/'cache'), HF_HOME=str(temp/'huggingface'))
        env.pop('ENSUREPIP_OPTIONS', None)
        return env

    def _run(self, command, work, log, timeout=1800):
        result = subprocess.run(command, cwd=self.root, env=self._environment(work), capture_output=True,
            text=True, encoding='utf-8', errors='replace', timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self._checked(log).write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('运行环境验证或安装失败：' + (result.stderr or result.stdout)[-2000:])
        return result

    def _stage_plan(self,state,records,wheels):
        """Record exact wheel/installer paths before pip can leave a partial prefix."""
        files=set()
        for name,wheel in zip(NAMES,wheels):
            info=name+'-'+records[name]['version']+'.dist-info'
            with zipfile.ZipFile(self._checked(wheel)) as archive:
                for member in archive.infolist():
                    if member.is_dir():continue
                    parts=self._safe_relative(member.filename)
                    if stat.S_ISLNK(member.external_attr>>16):raise ValueError('Linked wheel members are not supported')
                    if parts[0].endswith('.data'):
                        if len(parts)<3 or parts[1] not in {'purelib','platlib','scripts'}:
                            raise ValueError('Unsupported wheel installation path')
                        relative=('/'.join(('Scripts',*parts[2:])) if parts[1]=='scripts' else
                                  '/'.join(('Lib','site-packages',*parts[2:])))
                    else:relative='Lib/site-packages/'+member.filename
                    files.add(relative)
                for generated in ['INSTALLER','REQUESTED','RECORD']:
                    files.add('Lib/site-packages/'+info+'/'+generated)
                entry_points=info+'/entry_points.txt'
                if entry_points in archive.namelist():
                    if archive.getinfo(entry_points).file_size>65536:raise ValueError('Invalid wheel entry points')
                    config=configparser.ConfigParser(interpolation=None)
                    config.read_string(archive.read(entry_points).decode('utf-8'))
                    if config.has_section('gui_scripts'):raise ValueError('Unexpected wheel GUI entry points')
                    for script in config.options('console_scripts') if config.has_section('console_scripts') else []:
                        if script not in SCRIPT_NAMES:raise ValueError('Unexpected Torch entry point')
                        files.update('Scripts/'+script+suffix for suffix in ['.exe','-script.py',''])
        entries=sorted({'/'.join(PurePosixPath(path).parts[:2 if path.startswith('Scripts/') else 3]) for path in files})
        self._validate_entries(entries,state['targetRuntime'])
        plan=dict(schemaVersion=1,transactionId=state['transactionId'],runtime=state['targetRuntime'],entries=entries,files=sorted(files),complete=False)
        self._write(self._state_path(state['work'])/'stage-ownership.json',plan)

    def _record_stage(self,state,stage):
        entries=self._entries(stage,state['targetRuntime'])
        self._validate_entries(entries,state['targetRuntime'])
        self._write(self._state_path(state['work'])/'stage-ownership.json',dict(schemaVersion=1,
            transactionId=state['transactionId'],runtime=state['targetRuntime'],entries=entries,
            inventory=self._snapshot(stage,entries),complete=True))

    def _cleanup_stage(self,state,activated=False):
        stage=self._state_path(state['stage'])
        if not stage.exists():return
        if activated:
            self._remove_tree(stage,set())
            return
        plan=self._read(self._state_path(state['work'])/'stage-ownership.json')
        if (plan.get('schemaVersion')!=1 or plan.get('transactionId')!=state['transactionId']
                or plan.get('runtime')!=state['targetRuntime'] or type(plan.get('complete')) is not bool):
            raise ValueError('Staging cleanup lacks its verified ownership record')
        entries=plan.get('entries');self._validate_entries(entries,plan['runtime'])
        if plan['complete']:
            self._validate_inventory(plan.get('inventory'),entries)
            if plan.get('cleanupStarted'):
                expected={row['path']:row for row in plan['inventory']}
                remaining=self._snapshot(stage,[entry for entry in entries if (stage/entry).exists()])
                if any(expected.get(row['path'])!=row for row in remaining):
                    raise ValueError('Staging files changed during interrupted cleanup')
            else:
                self._verify_snapshot(stage,entries,plan['inventory'])
                plan['cleanupStarted']=True
                self._write(self._state_path(state['work'])/'stage-ownership.json',plan)
            files={row['path'] for row in plan['inventory']}
        else:
            paths=plan.get('files')
            if not isinstance(paths,list):raise ValueError('Invalid staging cleanup paths')
            for path in paths:
                self._safe_relative(path)
                if not any(path==entry or path.startswith(entry+'/') for entry in entries):
                    raise ValueError('Staging cleanup escaped its owned packages')
            files=set(paths)
        self._remove_tree(stage,files)

    def _cleanup_tmp(self,state):
        work=self._state_path(state['work']);directory=self._checked(work/'tmp')
        if not directory.exists():return
        rows=self._snapshot(work,['tmp'])
        self._remove_tree(directory,{row['path'][4:] for row in rows})

    def _cleanup_rolled_back_backup(self,state):
        if state['phase']!='rolled-back':return
        backup=self._state_path(state['backup'])
        if not backup.exists():return
        receipt_path=backup/'receipt.json'
        if receipt_path.exists():
            receipt=self._read(receipt_path)
            if (receipt.get('schemaVersion')!=1 or receipt.get('transactionId')!=state['transactionId']
                    or receipt.get('runtime')!=state['sourceRuntime'] or receipt.get('entries')!=state['oldEntries']
                    or receipt.get('inventory')!=state['oldInventory'] or receipt.get('bytes')!=state['retainedBytes']):
                raise ValueError('Rollback backup receipt changed; cleanup evidence was preserved')
        self._verify_snapshot(self.active,state['oldEntries'],state['oldInventory'])
        self._remove_tree(backup,{'receipt.json'})

    def _cleanup_failed(self,state):
        state.pop('cleanupWarning',None)
        warnings=[]
        for action in [lambda:self._cleanup_partials(state,current_only=True),lambda:self._cleanup_stage(state),
                       lambda:self._cleanup_tmp(state),lambda:self._cleanup_rolled_back_backup(state)]:
            try:action()
            except Exception as error:warnings.append(str(error))
        if warnings:state['cleanupWarning']='；'.join(warnings)

    def _stage(self, state, records, wheels):
        stage, work = self._state_path(state['stage']), self._state_path(state['work'])
        self._stage_plan(state,records,wheels)
        self._mkdir(stage)
        requirements = work / 'requirements.txt'
        requirements.write_text('\n'.join(f"{records[n]['name']}=={records[n]['version']} --hash=sha256:{records[n]['wheel']['sha256']}" for n in NAMES)+'\n', encoding='utf-8')
        # ensurepip's official bundled pip also works when portable venvs omit pip.
        code = ("import importlib.util,ensurepip,pathlib,runpy,sys; "
                "present=importlib.util.find_spec('pip') is not None; "
                "wheel=pathlib.Path(ensurepip.__file__).parent/'_bundled'/('pip-'+ensurepip.version()+'-py3-none-any.whl'); "
                "sys.path.insert(0,str(wheel)) if not present else None; "
                "sys.argv=['pip']+sys.argv[1:];runpy.run_module('pip',run_name='__main__')")
        self._run([str(self._python()), '-I', '-B', '-c', code, '--isolated', 'install', '--no-index',
            '--find-links', str(wheels[0].parent), '--ignore-installed', '--no-deps', '--no-compile', '--no-cache-dir',
            '--only-binary=:all:', '--require-hashes', '--prefix', str(stage), '--report', str(work/'install-report.json'),
            '-r', str(requirements)], work, work/'install.log')
        return stage

    def _probe(self, prefix, runtime, work, require_gpu=True):
        output = self._checked(work / ('probe-' + uuid.uuid4().hex + '.json'))
        command = [str(self._python()), '-I', '-B', str(self.root/'tools/torch-runtime-switch.py'), '_probe',
                   '--prefix', self._relative(prefix), '--target', runtime, '--output', self._relative(output)]
        if require_gpu:
            command.append('--require-gpu')
        self._run(command, work, output.with_suffix('.log'), timeout=600)
        result = self._read(output)
        if not result.get('passed') or (require_gpu and not result.get('gpuValidated')):
            raise RuntimeError('匹配的运行环境未通过实际计算验证，原环境已保留。')
        return result

    def _entries(self, prefix, runtime):
        site = self._checked(prefix / 'Lib/site-packages')
        distributions = list(metadata.distributions(path=[str(site)]))
        selected = {}
        for distribution in distributions:
            name = distribution.metadata.get('Name', '').lower().replace('_', '-')
            if name in NAMES:
                if name in selected:
                    raise ValueError('Duplicate runtime distribution: ' + name)
                selected[name] = distribution
        if set(selected) != set(NAMES):
            raise ValueError('Both pinned Torch distributions must be present')
        entries = set()
        for name, distribution in selected.items():
            expected = self.manifest['runtimes'][runtime][name]
            if distribution.version != expected or not distribution.files:
                raise ValueError('Runtime package does not match its pinned variant: ' + name)
            allowed = PACKAGE_ROOTS | {f'{name}-{expected}.dist-info'}
            scripts = {point.name for point in distribution.entry_points if point.group == 'console_scripts'}
            if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', value) for value in scripts):
                raise ValueError('Invalid Torch console entry point')
            for file in distribution.files:
                path = self._checked(distribution.locate_file(file))
                relative = path.relative_to(prefix).as_posix()
                parts = PurePosixPath(relative).parts
                if parts[:2] == ('Lib', 'site-packages') and len(parts) >= 3 and parts[2] in allowed:
                    entry = '/'.join(parts[:3])
                elif len(parts) == 2 and parts[0] == 'Scripts' and any(parts[1] in {s+'.exe', s+'-script.py', s} for s in scripts):
                    entry = relative
                else:
                    raise ValueError('Torch RECORD path is outside the managed package scope: ' + relative)
                if not path.exists() and file.hash is not None:
                    raise ValueError('Missing owned runtime file: ' + relative)
                if (prefix / entry).exists():
                    entries.add(entry)
        # No top-level path may also belong to another installed distribution.
        for distribution in distributions:
            if distribution in selected.values():
                continue
            for file in distribution.files or []:
                path = Path(os.path.abspath(distribution.locate_file(file)))
                if path.is_relative_to(prefix):
                    rel = path.relative_to(prefix).as_posix()
                    if any(rel == entry or rel.startswith(entry+'/') for entry in entries):
                        raise ValueError('Runtime package path is shared with another distribution')
        return sorted(entries)

    def _snapshot(self, prefix, entries):
        rows = []
        for relative in entries:
            path = self._checked(prefix / relative)
            stack = [path]
            while stack:
                item = stack.pop()
                self._checked(item)
                info = item.lstat()
                if stat.S_ISDIR(info.st_mode):
                    stack.extend(item.iterdir())
                elif stat.S_ISREG(info.st_mode):
                    rows.append(dict(path=item.relative_to(prefix).as_posix(), size=info.st_size, mtime_ns=info.st_mtime_ns))
                else:
                    raise ValueError('Unsupported runtime file type')
        return sorted(rows, key=lambda row: row['path'])

    def _verify_snapshot(self, prefix, entries, expected):
        if self._snapshot(prefix, entries) != expected:
            raise ValueError('Retained runtime files changed; recovery evidence was preserved')

    def _move(self, source, target):
        source, target = self._checked(source), self._checked(target)
        if target.exists():
            raise ValueError('Runtime destination already exists: ' + str(target))
        self._mkdir(target.parent)
        source.rename(target)

    def _step(self, state, kind, relative, source, target):
        step = dict(kind=kind, path=relative, status='intent')
        state['steps'].append(step)
        self._save(state)
        self._move(source, target)
        step['status'] = 'done'
        self._save(state)

    def _rollback(self, state):
        state['phase'] = 'rolling-back'
        self._save(state)
        stage, backup = self._new_source(state), self._state_path(state['backup'])
        for kind in ['new', 'old']:
            for step in reversed(state['steps']):
                if step['kind'] != kind or step['status'] == 'rolled-back':
                    continue
                current = self._checked(self.active / step['path'])
                other = self._checked((stage if kind == 'new' else backup) / step['path'])
                source, target = (current, other) if kind == 'new' else (other, current)
                if source.exists() and not target.exists():
                    self._move(source, target)
                elif target.exists() and not source.exists():
                    pass  # An intent was not executed, or this rollback move already completed.
                else:
                    raise ValueError('Runtime recovery found conflicting or missing paths; no unknown files were removed')
                step['status'] = 'rolled-back'
                self._save(state)
        self._verify_snapshot(self.active, state['oldEntries'], state['oldInventory'])
        if state.get('reuseTransactionId'):
            receipt_path=self._new_source(state)/'receipt.json'
            receipt=self._read(receipt_path)
            if receipt.get('consumed'):
                if receipt.get('consumedBy')!=state['transactionId']:
                    raise ValueError('Retained runtime was consumed by another transaction')
                receipt['consumed']=False
                receipt.pop('consumedBy',None)
                receipt['bytes']=sum(row['size'] for row in receipt['inventory'])
                self._write(receipt_path,receipt)
            state['consumeReuseStatus']='rolled-back'
            self._save(state)
        self._probe(self.active, state['sourceRuntime'], self._state_path(state['work']), require_gpu=False)
        state['phase'] = 'rolled-back'
        self._save(state)

    def prepare(self, target):
        if target not in self.manifest['runtimes']:
            raise ValueError('Unknown target runtime')
        with self._lease():
            previous = self._state()
            if previous and previous['phase'] == 'active' and previous['targetRuntime'] == target:
                if previous.get('cleanupWarning'):
                    self._assert_idle();self._retry_success_cleanup(previous)
                return self.status()
            if previous and previous['phase'] not in COMMITTED | {'failed'}:
                raise RuntimeError('请先恢复或完成上次运行环境操作。')
            if previous and previous['phase'] == 'active':
                raise RuntimeError('请先选择保留或删除上次切换的旧版运行环境。')
            self._assert_idle()
            if previous and previous['phase'] in {'failed','rolled-back'}:
                self._cleanup_failed(previous);self._save(previous)
                if previous.get('cleanupWarning'):
                    raise RuntimeError('请先清理或恢复上次准备留下的文件：'+previous['cleanupWarning'])
            assessment = self.status()
            if assessment['currentRuntime'] == target:
                return assessment
            if assessment['status'] != 'change-required' or assessment['targetRuntime'] != target:
                raise RuntimeError('显卡环境已变化，请重新检测后选择匹配版本。')
            source = assessment['currentRuntime']
            reusable=self._matching_retained(target)
            transaction = uuid.uuid4().hex
            state = dict(schemaVersion=1, transactionId=transaction, sourceRuntime=source, targetRuntime=target,
                phase='downloading', stage=self._relative(self.managed/'staged'/transaction),
                backup=self._relative(self.managed/'retained'/transaction), work=self._relative(self.managed/'transactions'/transaction),
                steps=[], createdAt=_now(), ownerPid=os.getpid(), retainedBytes=0, downloadBytes=assessment['downloadBytes'])
            state.update(cachedBytes=assessment['cachedBytes'],downloadRemainingBytes=assessment['downloadRemainingBytes'])
            if reusable:
                state['reuseTransactionId']=reusable['transactionId']
                state['newSource']=self._relative(self.managed/'retained'/reusable['transactionId'])
            else:
                state['newSource']=state['stage']
            work = self._mkdir(self._state_path(state['work']))
            try:
                self._emit(state, 'downloading', '正在下载匹配的运行环境。')
                if shutil.disk_usage(self.root).free < state['downloadBytes']*4 + 512*1024**2:
                    raise RuntimeError('磁盘空间不足，原运行环境保持不变。')
                records = runtime_sbom(self.root, target)['packages']
                wheels = []
                if reusable:
                    stage=self._new_source(state)
                    receipt=self._read(stage/'receipt.json')
                    self._verify_snapshot(stage,receipt['entries'],receipt['inventory'])
                else:
                    for name in NAMES:
                        wheels.append(self._download(state, name, records[name]))
                        state.setdefault('cacheWheelFiles',[]).append(records[name]['wheel']['file'])
                        if name not in state.get('downloadCacheHits',[]):
                            state['completedDownloadBytes'] = state.get('completedDownloadBytes', 0)+records[name]['wheel']['sizeBytes']
                    self._emit(state, 'staging', '下载校验完成，正在准备独立的新环境。')
                    stage = self._stage(state, records, wheels)
                    self._record_stage(state,stage)
                self._emit(state, 'probing', '正在验证新环境，原环境保持可用。')
                self._probe(stage, target, work)
                self._probe(self.active, source, work, require_gpu=False)
                self._assert_idle()
                again = self._hardware()
                if not again or not all(self._supports(target, item['capability']) for item in again):
                    raise RuntimeError('显卡环境在下载期间发生变化，原环境保持不变，请重新检测。')
                state['oldEntries'] = self._entries(self.active, source)
                state['newEntries'] = self._entries(stage, target)
                state['oldInventory'] = self._snapshot(self.active, state['oldEntries'])
                state['newInventory'] = self._snapshot(stage, state['newEntries'])
                state['retainedBytes'] = sum(row['size'] for row in state['oldInventory'])
                self._emit(state, 'switching', '新环境验证通过，正在切换；旧版将保留以便回退。')
                backup = self._mkdir(self._state_path(state['backup']))
                for relative in state['oldEntries']:
                    self._step(state, 'old', relative, self.active/relative, backup/relative)
                for relative in state['newEntries']:
                    self._step(state, 'new', relative, stage/relative, self.active/relative)
                self._emit(state, 'verifying-active', '正在验证切换结果。')
                state['activeProbe'] = self._probe(self.active, target, work)
                receipt = dict(schemaVersion=1, transactionId=transaction, runtime=source, bytes=state['retainedBytes'],
                    entries=state['oldEntries'], inventory=state['oldInventory'], createdAt=_now(), kept=False)
                self._write(backup/'receipt.json', receipt)
                self._consume_reused_receipt(state)
                state['phase'] = 'active'
                state['error'] = None
                self._save(state)
            except BaseException as error:
                state['error'] = str(error)
                if state.get('steps'):
                    try:
                        self._rollback(state)
                    except BaseException as recovery:
                        state.update(phase='recovery-required', error=str(error)+'；恢复未完成：'+str(recovery))
                        self._save(state)
                        raise RuntimeError(state['error']) from error
                else:
                    state['phase'] = 'failed'
                self._cleanup_failed(state)
                self._save(state)
                raise
            # The durable activation commit is the rollback boundary. Nothing
            # below, including status serialization and a disconnected UI,
            # may roll back a successfully activated runtime.
            self._retry_success_cleanup(state,wheels)
            result=self.status()
            try:self.progress(result)
            except Exception as notification:
                state['notificationWarning']=str(notification)
                self._save(state)
            return result

    def recover(self, transaction_id=None):
        with self._lease():
            self._assert_idle()
            state = self._state()
            if not state:
                return self.status()
            if transaction_id is not None and transaction_id != state['transactionId']:
                raise ValueError('Runtime transaction changed; refresh before recovering')
            if state['phase'] in {'active','kept','deleted'}:
                self._retry_success_cleanup(state)
                return self.status()
            if state['phase']=='deleting':
                return self.status()
            if state['phase']=='rolled-back':
                self._cleanup_failed(state);self._save(state)
                return self.status()
            if state.get('steps'):
                try:
                    self._rollback(state)
                except BaseException as error:
                    state.update(phase='recovery-required', error=str(error))
                    self._save(state)
                    raise
            else:
                state.update(phase='failed', error='上次准备已中断，原运行环境保持不变，可重新下载。')
            self._cleanup_failed(state)
            self._save(state)
            return self.status()

    def _choice(self, transaction_id):
        if not TOKEN.fullmatch(transaction_id or ''):
            raise ValueError('A matching runtime transaction identifier is required')
        state = self._state()
        if not state or state['transactionId'] != transaction_id:
            backup=self._checked(self.managed/'retained'/transaction_id)
            receipt=self._read(backup/'receipt.json') if (backup/'receipt.json').exists() else self._deletion_record(transaction_id)
            if receipt is None:raise ValueError('The retained runtime is no longer available')
            runtime=receipt.get('runtime')
            if (receipt.get('schemaVersion')!=1 or receipt.get('transactionId')!=transaction_id or runtime not in self.manifest['runtimes']
                    or receipt.get('deleted') or receipt.get('consumed') or not (receipt.get('kept') or receipt.get('deleteConfirmed'))):
                raise ValueError('The retained runtime is no longer available')
            self._validate_entries(receipt.get('entries'),runtime)
            self._validate_inventory(receipt.get('inventory'),receipt['entries'])
            active=installed_runtime(self.root,self.manifest)
            if active not in self.manifest['runtimes']:
                raise ValueError('The active runtime must be healthy before old files can be removed')
            state=dict(transactionId=transaction_id,sourceRuntime=runtime,targetRuntime=active,
                phase='deleting' if receipt.get('deleteConfirmed') else 'kept',historical=True,
                deleteConfirmed=bool(receipt.get('deleteConfirmed')),
                backup=self._relative(backup),work=self._relative(self.managed/'transactions'/transaction_id),
                oldEntries=receipt['entries'],oldInventory=receipt['inventory'],activeProbe={'passed':True})
        if state['phase'] not in {'active','kept','deleting'}:
            raise ValueError('Runtime transaction changed; refresh before choosing what to retain')
        if installed_runtime(self.root, self.manifest) != state['targetRuntime'] or not state.get('activeProbe', {}).get('passed'):
            raise ValueError('The new runtime is not active and verified; the old runtime cannot be deleted')
        backup = self._state_path(state['backup'])
        if state['phase'] == 'deleting' and state.get('deleteConfirmed') and not (backup/'receipt.json').exists():
            receipt = self._deletion_record(transaction_id)
            if receipt is None:raise ValueError('Interrupted deletion lacks its confirmed ownership record')
        else:
            receipt = self._read(backup/'receipt.json')
        if (receipt.get('transactionId') != transaction_id or receipt.get('runtime') != state['sourceRuntime']
                or receipt.get('entries') != state['oldEntries'] or receipt.get('inventory') != state['oldInventory']):
            raise ValueError('Retained runtime receipt differs from the verified transaction')
        return state, backup, receipt

    def _deletion_path(self,transaction_id):
        if not TOKEN.fullmatch(transaction_id or ''):raise ValueError('Invalid deletion transaction identifier')
        return self._checked(self.managed/'transactions'/transaction_id/'delete.json')

    def _deletion_record(self,transaction_id):
        path=self._deletion_path(transaction_id)
        if not path.exists():return None
        item=self._read(path)
        runtime=item.get('runtime')
        if (item.get('schemaVersion')!=1 or item.get('transactionId')!=transaction_id or runtime not in self.manifest['runtimes']
                or item.get('deleteConfirmed') is not True or type(item.get('complete')) is not bool):
            raise ValueError('Invalid retained deletion record')
        self._validate_entries(item.get('entries'),runtime)
        self._validate_inventory(item.get('inventory'),item['entries'])
        if item.get('bytes')!=sum(row['size'] for row in item['inventory']):raise ValueError('Invalid deletion inventory size')
        return item

    def _consume_reused_receipt(self,state):
        if not state.get('reuseTransactionId'):return
        source=self._new_source(state)
        receipt=self._read(source/'receipt.json')
        if (receipt.get('runtime')!=state['targetRuntime'] or receipt.get('entries')!=state['newEntries']
                or receipt.get('inventory')!=state['newInventory'] or receipt.get('consumed')):
            raise ValueError('The reused runtime receipt changed before commit')
        state['consumeReuseStatus']='intent'
        self._save(state)
        receipt.update(consumed=True,consumedBy=state['transactionId'],bytes=0)
        self._write(source/'receipt.json',receipt)
        state['consumeReuseStatus']='done'
        self._save(state)

    def _cleanup_success(self, state, wheels=None):
        records=runtime_sbom(self.root,state['targetRuntime'])['packages']
        directory=self._checked(self.managed/'wheels'/state['targetRuntime'])
        expected={records[name]['wheel']['file']:records[name]['wheel'] for name in NAMES}
        names=state.get('cacheWheelFiles',[]) if wheels is None else [self._checked(path).name for path in wheels]
        if not isinstance(names,list) or any(name not in expected for name in names):
            raise ValueError('Unknown cached wheel cleanup ownership')
        actions=[lambda:self._write(self._state_path(state['work'])/'download-receipt.json',
            dict(runtime=state['targetRuntime'],packages={name:records[name]['wheel'] for name in NAMES},verified=True,
                 downloaded=bool(state.get('completedDownloadBytes')),cachedBytes=state.get('cachedBytes',0),
                 reusedTransactionId=state.get('reuseTransactionId')))]
        def remove_wheel(name):
            path=self._checked(directory/name)
            if path.exists():path.unlink()
            receipt=self._cache_receipt(state['targetRuntime'],expected[name])
            if receipt.exists():receipt.unlink()
        for name in names:
            actions.append(lambda name=name:remove_wheel(name))
        actions.extend([lambda:self._cleanup_partials(state,current_only=False),
                        lambda:self._cleanup_stage(state,activated=True),lambda:self._cleanup_tmp(state)])
        warnings=[]
        for action in actions:
            try:action()
            except Exception as error:warnings.append(str(error))
        if directory.exists() and not any(directory.iterdir()):
            try:directory.rmdir()
            except OSError as error:warnings.append(str(error))
        if warnings:raise RuntimeError('；'.join(warnings))

    def _retry_success_cleanup(self,state,wheels=None):
        state.pop('cleanupWarning',None)
        try:
            self._cleanup_success(state,wheels)
            if state['phase']=='deleted':
                deletion=self._deletion_record(state['transactionId'])
                if deletion:self._cleanup_deleted_tmp(state['transactionId'],deletion)
        except Exception as error:state['cleanupWarning']=str(error)
        self._save(state)

    def _cleanup_deleted_tmp(self,transaction_id,deletion):
        deletion.pop('cleanupWarning',None)
        try:self._cleanup_tmp(dict(work=self._relative(self.managed/'transactions'/transaction_id)))
        except Exception as error:deletion['cleanupWarning']=str(error)
        self._write(self._deletion_path(transaction_id),deletion)

    def _cleanup_partials(self,state,current_only):
        directory=self._checked(self.managed/'wheels'/state['targetRuntime'])
        if not directory.exists():return
        records=runtime_sbom(self.root,state['targetRuntime'])['packages']
        for name in NAMES:
            wheel=records[name]['wheel']
            pattern=re.compile(re.escape(wheel['file'])+r'\.([0-9a-f]{32})\.partial$')
            for path in directory.glob(wheel['file']+'.*.partial'):
                match=pattern.fullmatch(path.name)
                if not match or (current_only and match[1]!=state['transactionId']):continue
                self._checked(path)
                if not path.is_file() or path.stat().st_size>wheel['sizeBytes']:
                    raise ValueError('Unknown partial download was preserved')
                path.unlink()

    def keep_retained(self, transaction_id):
        with self._lease():
            self._assert_idle()
            state, backup, receipt = self._choice(transaction_id)
            if state['phase'] == 'deleting':
                raise ValueError('旧版清理已开始，不能把不完整的旧版标记为保留。')
            receipt['kept'] = True
            self._write(backup/'receipt.json', receipt)
            state['phase'] = 'kept'
            if not state.get('historical'):self._retry_success_cleanup(state)
            return self.status()

    def _remove_tree(self, prefix, allowed_files):
        prefix = self._checked(prefix)
        files, directories = [], []
        stack = [prefix]
        while stack:
            parent = stack.pop()
            self._checked(parent)
            directories.append(parent)
            for item in parent.iterdir():
                self._checked(item)
                if item.is_dir():
                    stack.append(item)
                else:
                    if item.relative_to(prefix).as_posix() not in allowed_files:
                        raise ValueError('Unexpected retained files; deletion was refused')
                    files.append(item)
        # A receipt survives package/child-directory cleanup; an external
        # tombstone also survives the final receipt/root-directory window.
        for item in files:
            if item==prefix/'receipt.json':continue
            item.unlink()
        for directory in sorted(directories, key=lambda item:-len(item.parts)):
            if directory==prefix:continue
            directory.rmdir()
        if (prefix/'receipt.json').exists():(prefix/'receipt.json').unlink()
        prefix.rmdir()

    def delete_retained(self, transaction_id, confirmed=False):
        if not confirmed:
            raise ValueError('请先明确确认删除旧版运行环境。')
        with self._lease():
            self._assert_idle()
            previous = self._state()
            deletion=self._deletion_record(transaction_id)
            if deletion and deletion['complete']:
                if (self.managed/'retained'/transaction_id).exists():raise ValueError('Unexpected files after completed deletion')
                if previous and previous['transactionId']==transaction_id and previous['phase']=='deleting':
                    previous['phase']='deleted';self._save(previous)
                self._cleanup_deleted_tmp(transaction_id,deletion)
                if previous and previous['transactionId']==transaction_id:self._retry_success_cleanup(previous)
                return self.status()
            if previous and previous['transactionId'] == transaction_id and previous['phase'] == 'deleted':
                self._retry_success_cleanup(previous)
                return self.status()
            state, backup, receipt = self._choice(transaction_id)
            self._probe(self.active, state['targetRuntime'], self._state_path(state['work']), require_gpu=True)
            if state['phase'] != 'deleting':
                self._verify_snapshot(backup, receipt['entries'], receipt['inventory'])
                deletion=dict(receipt,deleteConfirmed=True,complete=False)
                self._write(self._deletion_path(transaction_id),deletion)
                state.update(phase='deleting', deleteConfirmed=True)
                if state.get('historical'):
                    receipt['deleteConfirmed']=True
                    self._write(backup/'receipt.json',receipt)
                else:self._save(state)
            elif backup.exists():
                existing=[entry for entry in receipt['entries'] if (backup/entry).exists()]
                expected={row['path']:row for row in receipt['inventory']}
                if any(expected.get(row['path'])!=row for row in self._snapshot(backup,existing)):
                    raise ValueError('Retained files changed during interrupted deletion')
            if backup.exists():
                self._remove_tree(backup, {row['path'] for row in receipt['inventory']} | {'receipt.json'})
            deletion=self._deletion_record(transaction_id) or dict(receipt,deleteConfirmed=True,complete=False)
            deletion['complete']=True
            self._write(self._deletion_path(transaction_id),deletion)
            state['phase'] = 'deleted'
            self._cleanup_deleted_tmp(transaction_id,deletion)
            if not state.get('historical'):self._retry_success_cleanup(state)
            return self.status()


def probe(root, prefix, runtime, output, require_gpu=False):
    """Fresh-process native acceptance. Never call this in the directory mover."""
    from .gpu_runtime import validate_torch_runtime, verify_runtime_records
    manager = RuntimeSwitch(root)
    prefix, output = manager._state_path(prefix), manager._state_path(output)
    allowed_prefix = prefix==manager.active or any(prefix.parent==manager.managed/name and TOKEN.fullmatch(prefix.name) for name in ['staged','retained'])
    if not allowed_prefix:
        raise ValueError('Only an active or managed runtime prefix can be probed')
    if (output.parent.parent!=manager.managed/'transactions' or not TOKEN.fullmatch(output.parent.name)
            or not re.fullmatch(r'probe-[0-9a-f]{32}\.json',output.name)):
        raise ValueError('Probe output must stay inside its managed transaction')
    sys.path.insert(0, str(prefix/'Lib/site-packages'))
    import torch
    import torchvision
    result = validate_torch_runtime(root, torch, torchvision, runtime, package_environment=prefix)
    result['recordFiles'] = verify_runtime_records(root, environment=prefix)
    torch.set_num_threads(2)
    values = torch.ones(1, 3, 8, 8)
    layer = torch.nn.Conv2d(3, 4, 3, padding=1)
    loss = layer(values).square().mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite Torch acceptance result')
    torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([.5]), .5)
    result['gpuValidated'] = False
    if torch.cuda.is_available():
        from .gpu_runtime import validate_cuda_devices
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            if not manager._supports(runtime, capability):
                continue
            validate_cuda_devices(torch, [index])
            device = torch.device('cuda', index)
            layer=layer.to(device)
            layer.zero_grad(set_to_none=True)
            calculated=layer(values.to(device))
            if not torch.isfinite(calculated).all():
                raise ValueError('Nonfinite CUDA acceptance result')
            calculated.square().mean().backward()
            with torch.no_grad():
                for parameter in layer.parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError('CUDA backward acceptance failed')
                    parameter.add_(parameter.grad,alpha=-.001)
            torch.cuda.synchronize(index)
            result['gpuValidated'] = True
            break
    if require_gpu and not result['gpuValidated']:
        raise ValueError('The downloaded runtime could not execute on a matching GPU')
    result['passed'] = True
    manager._write(output, result)
    return result
