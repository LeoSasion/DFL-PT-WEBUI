"""Runtime switching is tested only in synthetic projects, never the workstation venv."""
import base64
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'_internal/DeepFaceLab'))
from core.torch_runtime_switch import RuntimeSwitch,assert_runtime_available,probe


def write_packages(prefix,runtime):
    site=prefix/'Lib/site-packages';site.mkdir(parents=True,exist_ok=True)
    for name,version in [('torch','2.9.1+'+runtime),('torchvision','0.24.1+'+runtime)]:
        files={name+'/__init__.py':('runtime='+repr(runtime)).encode()}
        if name=='torch':
            files.update({'torchgen/__init__.py':b'generator','functorch/__init__.py':b'functional',
                          '../../Scripts/torchrun.exe':b'synthetic console '+runtime.encode()})
        info=f'{name}-{version}.dist-info'
        files[info+'/METADATA']=f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'.encode()
        files[info+'/entry_points.txt']=b'[console_scripts]\ntorchrun = torch.run:main\n' if name=='torch' else b''
        rows=[]
        for relative,data in files.items():
            path=site/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            rows.append([relative,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='),str(len(data))])
        rows.append([info+'/RECORD','',''])
        with (site/info/'RECORD').open('w',encoding='utf-8',newline='') as stream:csv.writer(stream).writerows(rows)


class SyntheticSwitch(RuntimeSwitch):
    def __init__(self,root):
        super().__init__(root)
        self.capability=(6,1);self.busy=None;self.downloads=[];self.probes=[]

    def _hardware(self):return [dict(name='Synthetic GPU',capability=self.capability)]
    def _busy_reason(self):return self.busy
    def _download(self,state,name,item):
        self.downloads.append((state['targetRuntime'],name))
        directory=self._mkdir(self.managed/'wheels'/state['targetRuntime'])
        path=directory/item['wheel']['file'];path.write_bytes(b'synthetic wheel');return path
    def _stage(self,state,records,wheels):
        prefix=self._mkdir(self._state_path(state['stage']))
        write_packages(prefix,state['targetRuntime']);return prefix
    def _probe(self,prefix,runtime,work,require_gpu=True):
        self.probes.append((str(prefix),runtime,require_gpu))
        from core.gpu_runtime import verify_runtime_records
        self._entries(prefix,runtime)
        verify_runtime_records(self.root,environment=prefix)
        if require_gpu and not self._supports(runtime,self.capability):raise RuntimeError('synthetic GPU mismatch')
        return dict(passed=True,gpuValidated=require_gpu)


@pytest.fixture
def manager(tmp_path):
    root=tmp_path/'project';(root/'release').mkdir(parents=True)
    (root/'_internal/DeepFaceLab').mkdir(parents=True)
    (root/'_internal/DeepFaceLab/me.py').write_text('# synthetic project',encoding='utf-8')
    for name in ['torch-runtimes.json','python-sbom.json']:
        shutil.copy2(ROOT/'release'/name,root/'release'/name)
    write_packages(root/'.venv','cu128')
    shared=root/'.venv/Lib/site-packages/shared-dependency/data.txt';shared.parent.mkdir();shared.write_text('shared',encoding='utf-8')
    media=root/'workspace/model/tae.pt';media.parent.mkdir(parents=True);media.write_bytes(b'protected synthetic model')
    return SyntheticSwitch(root)


def current_bytes(manager):
    return {p.relative_to(manager.active).as_posix():p.read_bytes() for p in manager.active.rglob('*') if p.is_file()}


def test_status_is_read_only_and_download_is_not_automatic(manager):
    before=current_bytes(manager)
    status=manager.status()
    assert status['status']=='change-required' and status['targetRuntime']=='cu126' and status['downloadRequired']
    assert not manager.managed.exists() and not manager.downloads
    assert current_bytes(manager)==before


def test_success_preserves_shared_dependencies_and_requires_cleanup_choice(manager):
    before=current_bytes(manager)
    status=manager.prepare('cu126')
    assert status['status']=='cleanup-choice' and status['canDelete']
    assert len(manager.downloads)==2
    assert not (manager.managed/'staged'/status['transactionId']).exists()
    assert not (manager.managed/'wheels/cu126').exists()
    assert (manager.active/'Lib/site-packages/shared-dependency/data.txt').read_bytes()==before['Lib/site-packages/shared-dependency/data.txt']
    assert (manager.root/'workspace/model/tae.pt').read_bytes()==b'protected synthetic model'
    assert not (manager.active/'Lib/site-packages/torch-2.9.1+cu128.dist-info').exists()
    assert manager.prepare('cu126')['transactionId']==status['transactionId']


def test_keep_then_change_back_reuses_retained_package_without_download(manager):
    first=manager.prepare('cu126');manager.keep_retained(first['transactionId'])
    manager.capability=(12,0)
    assert manager.status()['downloadRequired'] is False
    downloads=list(manager.downloads)
    second=manager.prepare('cu128')
    assert manager.downloads==downloads
    assert second['status']=='cleanup-choice' and second['currentRuntime']=='cu128'
    assert len(second['retainedRuntimes'])==1 and second['retainedRuntimes'][0]['runtime']=='cu126'
    with pytest.raises(ValueError,match='no longer available'):
        manager.delete_retained(first['transactionId'],True)


def test_reused_variant_rollback_restores_both_active_and_retained(manager,monkeypatch):
    first=manager.prepare('cu126');manager.keep_retained(first['transactionId']);manager.capability=(12,0)
    before=current_bytes(manager);original=manager._probe
    def fail_active(prefix,runtime,*args,**kwargs):
        if prefix==manager.active and runtime=='cu128':raise RuntimeError('post-activation failure')
        return original(prefix,runtime,*args,**kwargs)
    monkeypatch.setattr(manager,'_probe',fail_active)
    with pytest.raises(RuntimeError,match='post-activation'):
        manager.prepare('cu128')
    assert current_bytes(manager)==before
    assert manager.status()['downloadRequired'] is False
    old=manager.managed/'retained'/first['transactionId']
    assert (old/'Lib/site-packages/torch-2.9.1+cu128.dist-info').exists()


def test_failed_download_or_probe_does_not_move_the_old_runtime(manager,monkeypatch):
    before=current_bytes(manager)
    monkeypatch.setattr(manager,'_download',lambda *args:(_ for _ in ()).throw(ValueError('bad wheel hash')))
    with pytest.raises(ValueError,match='bad wheel hash'):manager.prepare('cu126')
    assert current_bytes(manager)==before
    assert manager._state()['phase']=='failed'
    assert not manager.status()['canDelete']


def test_post_activation_failure_rolls_back_before_return(manager,monkeypatch):
    before=current_bytes(manager);original=manager._probe
    def fail(prefix,runtime,*args,**kwargs):
        if prefix==manager.active and runtime=='cu126':raise RuntimeError('new runtime failed')
        return original(prefix,runtime,*args,**kwargs)
    monkeypatch.setattr(manager,'_probe',fail)
    with pytest.raises(RuntimeError,match='new runtime failed'):manager.prepare('cu126')
    assert current_bytes(manager)==before
    assert manager._state()['phase']=='rolled-back'


def test_crash_during_move_can_be_recovered_from_persisted_intent(manager,monkeypatch):
    before=current_bytes(manager);original=manager._step;original_rollback=manager._rollback
    class Interrupted(BaseException):pass
    def step(state,kind,relative,source,target):
        original(state,kind,relative,source,target)
        if kind=='old':raise Interrupted()
    monkeypatch.setattr(manager,'_step',step)
    monkeypatch.setattr(manager,'_rollback',lambda state:(_ for _ in ()).throw(Interrupted()))
    with pytest.raises(RuntimeError):manager.prepare('cu126')
    state=manager._state();assert state['phase']=='recovery-required'
    monkeypatch.setattr(manager,'_rollback',original_rollback)
    assert manager.status()['canRecover']
    manager.recover(state['transactionId'])
    assert current_bytes(manager)==before


def test_busy_state_blocks_every_mutating_operation(manager):
    before=current_bytes(manager);manager.busy='synthetic task active'
    assert not manager.status()['canSwitch']
    with pytest.raises(RuntimeError,match='task active'):manager.prepare('cu126')
    assert current_bytes(manager)==before and not manager.downloads


def test_delete_requires_matching_confirmation_and_never_touches_active(manager):
    status=manager.prepare('cu126');token=status['transactionId'];before=current_bytes(manager)
    with pytest.raises(ValueError,match='确认'):manager.delete_retained(token)
    with pytest.raises((ValueError,FileNotFoundError)):manager.delete_retained('f'*32,True)
    manager.delete_retained(token,True)
    assert current_bytes(manager)==before
    assert not (manager.managed/'retained'/token).exists()
    assert manager.delete_retained(token,True)['status']=='compatible'


def test_valid_historical_retained_receipt_can_be_deleted_after_a_failed_retry(manager,monkeypatch):
    first=manager.prepare('cu126');manager.keep_retained(first['transactionId']);manager.capability=(12,0)
    monkeypatch.setattr(manager,'_probe',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('interrupted validation')))
    with pytest.raises(RuntimeError):manager.prepare('cu128')
    monkeypatch.setattr(manager,'_probe',lambda *a,**k:dict(passed=True,gpuValidated=True))
    manager.capability=(6,1)
    assert manager._state()['transactionId']!=first['transactionId']
    latest=manager._state()['transactionId']
    manager.delete_retained(first['transactionId'],True)
    assert manager._state()['transactionId']==latest
    assert not (manager.managed/'retained'/first['transactionId']).exists()


def test_interrupted_deletion_can_resume_without_receipt_and_with_inventory_subset(manager,monkeypatch):
    status=manager.prepare('cu126');token=status['transactionId'];original=manager._remove_tree
    def interrupted(prefix,allowed):
        (prefix/'receipt.json').unlink()
        (prefix/'Lib/site-packages/torch/__init__.py').unlink()
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(manager,'_remove_tree',interrupted)
    with pytest.raises(RuntimeError):manager.delete_retained(token,True)
    assert manager._state()['phase']=='deleting'
    assert_runtime_available(manager.root)
    assert manager.status()['status']=='cleanup-choice'
    monkeypatch.setattr(manager,'_remove_tree',original)
    manager.delete_retained(token,True)
    assert not (manager.managed/'retained'/token).exists()


@pytest.mark.parametrize('historical',[False,True])
def test_final_retained_directory_failure_keeps_a_visible_retryable_tombstone(manager,monkeypatch,historical):
    first=manager.prepare('cu126');token=first['transactionId']
    if historical:
        manager.keep_retained(token);manager.capability=(12,0)
        original_probe=manager._probe
        monkeypatch.setattr(manager,'_probe',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('failed later prepare')))
        with pytest.raises(RuntimeError):manager.prepare('cu128')
        monkeypatch.setattr(manager,'_probe',original_probe);manager.capability=(6,1)
    latest=manager._state()['transactionId'];before=current_bytes(manager)
    backup=manager.managed/'retained'/token;original_rmdir=Path.rmdir
    def rmdir(path):
        if path==backup:raise PermissionError('retained root temporarily busy')
        return original_rmdir(path)
    monkeypatch.setattr(Path,'rmdir',rmdir)
    with pytest.raises(PermissionError,match='temporarily busy'):manager.delete_retained(token,True)
    assert backup.is_dir() and not (backup/'receipt.json').exists()
    assert not manager._deletion_record(token)['complete']
    retained=manager.status()['retainedRuntimes']
    assert any(row['transactionId']==token and row['deleting'] and not row['canReuse'] for row in retained)
    assert current_bytes(manager)==before
    assert_runtime_available(manager.root)
    monkeypatch.setattr(Path,'rmdir',original_rmdir)
    manager.delete_retained(token,True)
    assert not backup.exists() and manager._deletion_record(token)['complete']
    assert manager._state()['transactionId']==latest and current_bytes(manager)==before
    manager.delete_retained(token,True)


@pytest.mark.parametrize('historical',[False,True])
def test_deletion_commit_interruption_after_directory_removal_remains_visible(manager,monkeypatch,historical):
    first=manager.prepare('cu126');token=first['transactionId']
    if historical:
        manager.keep_retained(token);manager.capability=(12,0);original_probe=manager._probe
        monkeypatch.setattr(manager,'_probe',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('failed later prepare')))
        with pytest.raises(RuntimeError):manager.prepare('cu128')
        monkeypatch.setattr(manager,'_probe',original_probe);manager.capability=(6,1)
    latest=manager._state()['transactionId'];before=current_bytes(manager);original_write=manager._write
    def write(path,value):
        if path==manager._deletion_path(token) and value.get('complete'):raise OSError('deletion commit interrupted')
        return original_write(path,value)
    monkeypatch.setattr(manager,'_write',write)
    with pytest.raises(OSError,match='commit interrupted'):manager.delete_retained(token,True)
    assert not (manager.managed/'retained'/token).exists()
    assert any(row['transactionId']==token and row['deleting'] for row in manager.status()['retainedRuntimes'])
    assert_runtime_available(manager.root)
    monkeypatch.setattr(manager,'_write',original_write)
    manager.delete_retained(token,True)
    assert manager._deletion_record(token)['complete']
    assert manager._state()['transactionId']==latest and current_bytes(manager)==before


def test_unknown_or_changed_retained_files_are_not_deleted(manager):
    status=manager.prepare('cu126');token=status['transactionId'];backup=manager.managed/'retained'/token
    (backup/'unexpected.txt').write_text('do not delete',encoding='utf-8')
    with pytest.raises(ValueError,match='Unexpected'):manager.delete_retained(token,True)
    assert (backup/'unexpected.txt').exists()
    assert (backup/'Lib/site-packages/torch/__init__.py').exists()


@pytest.mark.parametrize('bad', ['../workspace','Lib/site-packages/../../workspace','/workspace','Scripts/../python.exe','Lib\\site-packages\\torch','Lib/site-packages/unrelated'])
def test_tampered_journal_entries_are_rejected_before_recovery(manager,bad):
    status=manager.prepare('cu126');state=manager._state();state['phase']='recovery-required'
    state['oldEntries']=[bad];manager._save(state)
    protected=(manager.root/'workspace/model/tae.pt').read_bytes()
    with pytest.raises(ValueError):manager.recover(status['transactionId'])
    assert (manager.root/'workspace/model/tae.pt').read_bytes()==protected


def test_tampered_step_or_inventory_is_rejected(manager):
    manager.prepare('cu126');state=manager._state();state['steps'][0]['path']='../workspace/model'
    manager._save(state)
    with pytest.raises(ValueError,match='move record'):manager.status()


def test_private_probe_cannot_write_to_workspace(manager):
    protected=manager.root/'workspace/model/tae.pt';before=protected.read_bytes()
    with pytest.raises(ValueError,match='Probe output'):
        probe(manager.root,'.venv','cu128','workspace/model/tae.pt')
    assert protected.read_bytes()==before


def test_dead_download_owner_is_recoverable_without_changing_packages(manager,monkeypatch):
    monkeypatch.setattr(manager,'_download',lambda *a:(_ for _ in ()).throw(RuntimeError('offline')))
    with pytest.raises(RuntimeError):manager.prepare('cu126')
    state=manager._state();state.update(phase='downloading',ownerPid=2147483647);manager._save(state)
    assert manager.status()['canRecover']
    with pytest.raises(RuntimeError,match='尚未完成'):assert_runtime_available(manager.root)
    manager.recover(state['transactionId']);assert_runtime_available(manager.root)


def test_shared_guard_blocks_a_live_cross_process_operation(manager):
    with manager._lease():
        with pytest.raises(RuntimeError,match='正在'):assert_runtime_available(manager.root)
    assert_runtime_available(manager.root)


def test_python_repair_guard_requires_a_matching_private_token_and_live_source_lock(manager):
    import msvcrt
    directory=manager.root/'.launcher-install/source';directory.mkdir(parents=True)
    token='a'*32;lock_token='b'*32
    state=dict(schemaVersion=1,transactionId=token,phase='installing',hadVenv=True,lockToken=lock_token,lockOwnerPid=os.getpid(),
        backup='.launcher-install/source/venv-backup-'+token,session='.launcher-install/source/python-session-'+token)
    manager._write(directory/'python-repair.json',state)
    manager._write(directory/'source-install-owner.json',dict(schemaVersion=1,token=lock_token,pid=os.getpid()))
    with (directory/'source-install.lock').open('w+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with pytest.raises(RuntimeError,match='Python'):assert_runtime_available(manager.root)
            assert_runtime_available(manager.root,token)
            with pytest.raises(RuntimeError,match='修复记录'):assert_runtime_available(manager.root,'c'*32)
            with manager._lease():
                with pytest.raises(RuntimeError,match='正在'):assert_runtime_available(manager.root,token)
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    with pytest.raises(RuntimeError,match='Python'):assert_runtime_available(manager.root)
    with pytest.raises(RuntimeError,match='修复记录'):assert_runtime_available(manager.root,token)
    state['phase']='committed';state['validated']=True;manager._write(directory/'python-repair.json',state)
    assert_runtime_available(manager.root)
    state['backup']='../workspace/model';manager._write(directory/'python-repair.json',state)
    with pytest.raises(RuntimeError,match='修复记录'):assert_runtime_available(manager.root)


def test_download_hash_failure_preserves_old_environment(manager,monkeypatch):
    before=current_bytes(manager)
    class Response:
        url='https://download.pytorch.org/pinned.whl'
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,size):return b'bad' if not getattr(self,'done',False) and not setattr(self,'done',True) else b''
    monkeypatch.setattr('core.torch_runtime_switch.urllib.request.urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(manager,'_download',RuntimeSwitch._download.__get__(manager))
    with pytest.raises(ValueError,match='SHA-256'):manager.prepare('cu126')
    assert current_bytes(manager)==before
    assert not list((manager.managed/'wheels/cu126').glob('*.partial'))


def test_cache_cleanup_failure_does_not_rollback_a_committed_runtime(manager,monkeypatch):
    monkeypatch.setattr(manager,'_cleanup_success',lambda *a:(_ for _ in ()).throw(PermissionError('cache is busy')))
    status=manager.prepare('cu126')
    assert status['status']=='cleanup-choice' and status['currentRuntime']=='cu126'
    assert manager._state()['phase']=='active' and 'cache is busy' in manager._state()['cleanupWarning']


def test_hardware_changes_again_before_cleanup_old_matching_runtime_is_preserved(manager):
    status=manager.prepare('cu126');token=status['transactionId'];manager.capability=(12,0)
    assert not manager.status()['canDelete']
    with pytest.raises(RuntimeError,match='GPU mismatch'):manager.delete_retained(token,True)
    assert (manager.managed/'retained'/token/'Lib/site-packages/torch-2.9.1+cu128.dist-info').exists()


def test_committed_reuse_does_not_rollback_when_ui_notification_breaks(manager):
    first=manager.prepare('cu126');manager.keep_retained(first['transactionId']);manager.capability=(12,0)
    def progress(value):
        if value['status']=='cleanup-choice':raise BrokenPipeError('UI closed')
    manager.progress=progress
    second=manager.prepare('cu128')
    assert second['currentRuntime']=='cu128' and manager._state()['phase']=='active'
    assert manager._state()['notificationWarning']=='UI closed'
    assert manager._read(manager.managed/'retained'/first['transactionId']/'receipt.json')['consumed']


def test_early_cache_failure_cannot_resurrect_an_empty_consumed_runtime(manager,monkeypatch):
    first=manager.prepare('cu126');manager.keep_retained(first['transactionId']);manager.capability=(12,0)
    original=manager._cleanup_success
    monkeypatch.setattr(manager,'_cleanup_success',lambda *a:(_ for _ in ()).throw(OSError('cleanup unavailable')))
    second=manager.prepare('cu128');manager.keep_retained(second['transactionId'])
    monkeypatch.setattr(manager,'_cleanup_success',original);manager.capability=(6,1)
    third=manager.prepare('cu126');manager.keep_retained(third['transactionId'])
    manager.delete_retained(third['transactionId'],True)
    manager.capability=(12,0)
    assert manager.status()['downloadRequired'] is True
    downloads=len(manager.downloads);manager.prepare('cu128')
    assert len(manager.downloads)==downloads+2


def test_failed_activation_commit_restores_consumed_receipt_during_rollback(manager,monkeypatch):
    first=manager.prepare('cu126');manager.keep_retained(first['transactionId']);manager.capability=(12,0)
    original=manager._save
    def save(state):
        if state['phase']=='active':raise OSError('commit unavailable')
        return original(state)
    monkeypatch.setattr(manager,'_save',save)
    with pytest.raises(OSError,match='commit unavailable'):manager.prepare('cu128')
    receipt=manager._read(manager.managed/'retained'/first['transactionId']/'receipt.json')
    assert not receipt.get('consumed') and receipt['bytes']>0 and receipt['kept']
    assert manager.status()['downloadRequired'] is False


def test_success_removes_only_known_historical_partial_downloads(manager):
    directory=manager._mkdir(manager.managed/'wheels/cu126')
    record=json.loads((manager.root/'release/torch-runtimes.json').read_text(encoding='utf-8'))['runtimes']['cu126']['packages']['torch']['wheel']
    partial=directory/(record['file']+'.'+('f'*32)+'.partial');partial.write_bytes(b'incomplete')
    unknown=directory/'user-note.txt';unknown.write_bytes(b'keep')
    manager.prepare('cu126')
    assert not partial.exists() and unknown.read_bytes()==b'keep'


def test_runtime_busy_regex_matches_actual_windows_process_names(manager,monkeypatch):
    commands=[];options=[]
    def run(command,**kwargs):
        commands.append(command);options.append(kwargs)
        return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr('core.torch_runtime_switch.subprocess.run',run)
    assert manager._busy_reason() is None
    # The fixture overrides the method; exercise production construction directly.
    RuntimeSwitch._busy_reason(manager)
    code=commands[-1][-1]
    assert r"\.exe$'" in code and r"\\.exe$'" not in code
    assert '$_.CommandLine' in code and str(os.getpid()) in code
    assert options[-1]['encoding']=='utf-8' and options[-1]['errors']=='replace'


def test_runtime_busy_regex_is_valid_in_real_windows_regex_engine(manager,monkeypatch):
    commands=[];actual=subprocess.run
    def run(command,**kwargs):
        commands.append(command);return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr('core.torch_runtime_switch.subprocess.run',run)
    RuntimeSwitch._busy_reason(manager)
    code=commands[-1][-1]
    import re
    pattern=re.search(r"-match '([^']+)'",code).group(1)
    result=actual(['powershell.exe','-NoProfile','-NonInteractive','-Command',
        "$r=[regex]'"+pattern+"'; if(-not $r.IsMatch('python.exe') -or -not $r.IsMatch('node.exe') -or $r.IsMatch('pythonXexe')){exit 9}; 'passed'"],
        capture_output=True,text=True,timeout=20)
    assert result.returncode==0 and 'passed' in result.stdout,result.stdout+result.stderr


def test_real_busy_probe_detects_an_external_python_running_this_project(manager):
    # Interpreter lives outside the synthetic project; its argument points inside.
    script=manager.root/'sleep_for_runtime_test.py'
    script.write_text('import time\ntime.sleep(20)\n',encoding='utf-8')
    child=subprocess.Popen([sys.executable,'-I','-B',str(script)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        assert RuntimeSwitch._busy_reason(manager) is not None
    finally:
        child.terminate();child.wait(timeout=10)


def test_retained_directory_link_is_never_followed_or_deleted(manager,tmp_path):
    status=manager.prepare('cu126');backup=manager.managed/'retained'/status['transactionId']
    outside=tmp_path/'outside';outside.mkdir();sentinel=outside/'tae.pt';sentinel.write_bytes(b'synthetic protected model')
    link=backup/'Lib/site-packages/torch/external'
    try:link.symlink_to(outside,target_is_directory=True)
    except OSError:
        result=subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(link),str(outside)],capture_output=True,text=True)
        if result.returncode:pytest.skip('Windows test cannot create a symbolic link or junction: '+result.stderr)
    with pytest.raises(ValueError,match='Linked'):manager.delete_retained(status['transactionId'],True)
    assert sentinel.read_bytes()==b'synthetic protected model' and (link.is_symlink() or link.is_junction())
    os.rmdir(link) if link.is_junction() else link.unlink()


def test_corrupt_committed_journal_is_not_treated_as_available(manager):
    manager.prepare('cu126');state=manager._state();state['oldEntries']=[];state['oldInventory']=[];state['steps']=[]
    manager._save(state)
    with pytest.raises(RuntimeError,match='无法确认'):assert_runtime_available(manager.root)


def test_initial_notification_failure_is_a_clean_failed_transaction(manager):
    before=current_bytes(manager)
    manager.progress=lambda value:(_ for _ in ()).throw(BrokenPipeError('initial UI disconnected'))
    with pytest.raises(BrokenPipeError):manager.prepare('cu126')
    assert manager._state()['phase']=='failed' and not manager.status()['canRecover']
    assert current_bytes(manager)==before


@pytest.mark.parametrize('unknown',[False,True])
def test_failed_activation_commit_cleans_its_empty_backup_receipt_after_rollback(manager,monkeypatch,unknown):
    before=current_bytes(manager);original_save=manager._save
    def save(state):
        if state['phase']=='active':
            if unknown:(manager._state_path(state['backup'])/'user-added.txt').write_bytes(b'preserve evidence')
            raise PermissionError('activation commit failed')
        return original_save(state)
    monkeypatch.setattr(manager,'_save',save)
    with pytest.raises(PermissionError):manager.prepare('cu126')
    state=manager._state();backup=manager._state_path(state['backup'])
    assert state['phase']=='rolled-back' and current_bytes(manager)==before
    if unknown:
        assert (backup/'user-added.txt').read_bytes()==b'preserve evidence'
        assert 'Unexpected' in state['cleanupWarning']
    else:
        assert not backup.exists() and manager.status()['retainedRuntimes']==[]


def test_atomic_json_failure_removes_only_its_own_temporary_file(manager,monkeypatch):
    path=manager.managed/'test.json';manager._mkdir(manager.managed)
    unknown=manager.managed/'test.json.unknown.tmp';unknown.write_bytes(b'keep')
    monkeypatch.setattr('core.torch_runtime_switch.os.replace',lambda *a:(_ for _ in ()).throw(OSError('replace failed')))
    with pytest.raises(OSError,match='replace failed'):manager._write(path,{'value':1})
    assert list(manager.managed.glob('test.json.*.tmp'))==[unknown]


def test_probe_failure_removes_owned_staging_and_scratch_but_preserves_wheels(manager,monkeypatch):
    original=manager._probe;before=current_bytes(manager)
    def fail(prefix,runtime,work,**kwargs):
        scratch=manager._mkdir(work/'tmp/cuda-cache');(scratch/'kernel.bin').write_bytes(b'generated')
        if prefix!=manager.active:raise RuntimeError('synthetic native probe failure')
        return original(prefix,runtime,work,**kwargs)
    monkeypatch.setattr(manager,'_probe',fail)
    with pytest.raises(RuntimeError,match='native probe failure'):manager.prepare('cu126')
    state=manager._state()
    assert not manager._state_path(state['stage']).exists()
    assert not (manager._state_path(state['work'])/'tmp').exists()
    assert len(list((manager.managed/'wheels/cu126').glob('*.whl')))==2
    assert current_bytes(manager)==before


@pytest.mark.parametrize('action',['recover','keep','delete'])
def test_post_commit_cache_failure_cleans_other_artifacts_and_public_action_retries(manager,monkeypatch,action):
    original_unlink=Path.unlink;original_probe=manager._probe
    def probe_with_scratch(prefix,runtime,work,**kwargs):
        (manager._mkdir(work/'tmp/cuda-cache')/'kernel.bin').write_bytes(b'generated probe cache')
        return original_probe(prefix,runtime,work,**kwargs)
    monkeypatch.setattr(manager,'_probe',probe_with_scratch)
    def unlink(path,*args,**kwargs):
        if path.suffix=='.whl':raise PermissionError('cache wheel temporarily busy')
        return original_unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',unlink)
    status=manager.prepare('cu126');state=manager._state();token=state['transactionId'];before=current_bytes(manager)
    assert status['canRetryCleanup'] and not status['canRecover']
    assert state['phase']=='active' and not manager._state_path(state['stage']).exists()
    assert not (manager._state_path(state['work'])/'tmp').exists()
    assert_runtime_available(manager.root)
    monkeypatch.setattr(Path,'unlink',original_unlink)
    if action=='recover':result=manager.recover(token)
    elif action=='keep':result=manager.keep_retained(token)
    else:result=manager.delete_retained(token,True)
    assert not result['canRetryCleanup'] and not result['cleanupWarning']
    assert not (manager.managed/'wheels/cu126').exists()
    assert not (manager._state_path(state['work'])/'tmp').exists()
    assert current_bytes(manager)==before


def test_failed_cleanup_keeps_its_token_until_explicit_recovery_succeeds(manager,monkeypatch):
    original_probe=manager._probe
    def fail(prefix,runtime,work,**kwargs):
        (prefix/'Lib/site-packages/torch/user-note.txt').write_bytes(b'preserve unknown')
        raise RuntimeError('probe failed')
    monkeypatch.setattr(manager,'_probe',fail)
    with pytest.raises(RuntimeError,match='probe failed'):manager.prepare('cu126')
    token=manager._state()['transactionId'];downloads=list(manager.downloads)
    assert manager.status()['canRecover']
    with pytest.raises(RuntimeError,match='上次准备'):manager.prepare('cu126')
    assert manager._state()['transactionId']==token and manager.downloads==downloads
    stage=manager._state_path(manager._state()['stage'])
    (stage/'Lib/site-packages/torch/user-note.txt').unlink()
    monkeypatch.setattr(manager,'_probe',original_probe)
    recovered=manager.recover(token)
    assert not recovered['canRecover'] and not recovered['cleanupWarning'] and not stage.exists()
    assert manager.prepare('cu126')['transactionId']!=token


@pytest.mark.parametrize('changed',[False,True])
def test_owned_stage_cleanup_can_resume_after_partial_unlink_and_rejects_changes(manager,monkeypatch,changed):
    monkeypatch.setattr(manager,'_probe',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('probe failed')))
    original_unlink=Path.unlink;calls=[]
    def unlink(path,*args,**kwargs):
        if 'staged' in path.parts and 'site-packages' in path.parts:
            calls.append(path)
            if len(calls)==2:raise PermissionError('second staging file busy')
        return original_unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',unlink)
    with pytest.raises(RuntimeError,match='probe failed'):manager.prepare('cu126')
    state=manager._state();stage=manager._state_path(state['stage']);token=state['transactionId']
    assert stage.exists() and not calls[0].exists() and calls[1].exists()
    assert manager._read(manager._state_path(state['work'])/'stage-ownership.json')['cleanupStarted']
    monkeypatch.setattr(Path,'unlink',original_unlink)
    if changed:calls[1].write_bytes(b'modified while cleanup interrupted')
    result=manager.recover(token)
    if changed:
        assert result['canRecover'] and 'changed' in result['cleanupWarning'] and calls[1].exists()
    else:
        assert not stage.exists() and not result['cleanupWarning'] and not result['canRecover']


def test_completed_historical_delete_retries_its_scratch_cleanup(manager,monkeypatch):
    first=manager.prepare('cu126');token=first['transactionId'];manager.keep_retained(token)
    manager.capability=(12,0);original_probe=manager._probe
    monkeypatch.setattr(manager,'_probe',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('later prepare failed')))
    with pytest.raises(RuntimeError):manager.prepare('cu128')
    monkeypatch.setattr(manager,'_probe',original_probe);manager.capability=(6,1)
    work=manager.managed/'transactions'/token;scratch=manager._mkdir(work/'tmp');(scratch/'cache').write_bytes(b'generated')
    original_cleanup=manager._cleanup_tmp
    monkeypatch.setattr(manager,'_cleanup_tmp',lambda state:(_ for _ in ()).throw(PermissionError('scratch busy')))
    manager.delete_retained(token,True)
    row=next(row for row in manager.status()['retainedRuntimes'] if row['transactionId']==token)
    assert row['deleting'] and row['canDelete'] and not row['canReuse'] and row['bytes']==0
    monkeypatch.setattr(manager,'_cleanup_tmp',original_cleanup)
    manager.delete_retained(token,True)
    assert not scratch.exists() and not any(row['transactionId']==token for row in manager.status()['retainedRuntimes'])


@pytest.mark.parametrize('unknown',[False,True])
def test_partial_pip_prefix_cleanup_uses_exact_wheel_ownership(manager,monkeypatch,unknown):
    before=current_bytes(manager)
    def download(state,name,item):
        path=manager._mkdir(manager.managed/'wheels'/state['targetRuntime'])/item['wheel']['file']
        info=name+'-'+item['version']+'.dist-info'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr(name+'/__init__.py',b'official synthetic source')
            archive.writestr(info+'/METADATA',b'synthetic metadata')
        return path
    def install(command,work,log,**kwargs):
        prefix=Path(command[command.index('--prefix')+1]);package=prefix/'Lib/site-packages/torch'
        package.mkdir(parents=True);(package/'__init__.py').write_bytes(b'partial')
        if unknown:(package/'user-added.txt').write_bytes(b'preserve unknown')
        (manager._mkdir(work/'tmp')/'pip.tmp').write_bytes(b'scratch')
        raise RuntimeError('pip interrupted')
    monkeypatch.setattr(manager,'_download',download)
    monkeypatch.setattr(manager,'_stage',RuntimeSwitch._stage.__get__(manager))
    monkeypatch.setattr(manager,'_python',lambda:Path(sys.executable))
    monkeypatch.setattr(manager,'_run',install)
    with pytest.raises(RuntimeError,match='pip interrupted'):manager.prepare('cu126')
    state=manager._state();stage=manager._state_path(state['stage'])
    if unknown:
        assert (stage/'Lib/site-packages/torch/user-added.txt').read_bytes()==b'preserve unknown'
        assert 'Unexpected' in state['cleanupWarning']
    else:assert not stage.exists()
    assert not (manager._state_path(state['work'])/'tmp').exists()
    assert current_bytes(manager)==before


def test_explicit_recovery_cleans_only_the_interrupted_token_partial_and_stage(manager,monkeypatch):
    original=manager._cleanup_failed
    monkeypatch.setattr(manager,'_cleanup_failed',lambda state:None)
    monkeypatch.setattr(manager,'_probe',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('interrupted probe')))
    with pytest.raises(RuntimeError):manager.prepare('cu126')
    state=manager._state();state.update(phase='probing',ownerPid=2147483647);manager._save(state)
    wheel=json.loads((manager.root/'release/torch-runtimes.json').read_text())['runtimes']['cu126']['packages']['torch']['wheel']
    directory=manager._mkdir(manager.managed/'wheels/cu126')
    own=directory/(wheel['file']+'.'+state['transactionId']+'.partial');own.write_bytes(b'partial')
    other=directory/(wheel['file']+'.'+('f'*32)+'.partial');other.write_bytes(b'other')
    scratch=manager._mkdir(manager._state_path(state['work'])/'tmp');(scratch/'scratch').write_bytes(b'cache')
    monkeypatch.setattr(manager,'_cleanup_failed',original)
    manager.recover(state['transactionId']);manager.recover(state['transactionId'])
    assert not own.exists() and other.read_bytes()==b'other'
    assert not manager._state_path(state['stage']).exists() and not scratch.exists()


def test_verified_wheel_cache_avoids_network_and_status_avoids_hashing(manager,monkeypatch):
    from core.gpu_runtime import runtime_sbom
    data=b'local synthetic wheel bytes'
    records=runtime_sbom(manager.root,'cu126')['packages']
    for name in ['torch','torchvision']:
        wheel=records[name]['wheel'];wheel.update(sizeBytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        path=manager._mkdir(manager.managed/'wheels/cu126')/wheel['file'];path.write_bytes(data)
        manager._record_cached_wheel('cu126',wheel,path)
    monkeypatch.setattr('core.torch_runtime_switch.runtime_sbom',lambda *a:dict(packages=records))
    monkeypatch.setattr(manager,'_digest',lambda *a:(_ for _ in ()).throw(AssertionError('status read wheel contents')))
    status=manager.status()
    assert status['cachedBytes']==len(data)*2 and status['downloadRemainingBytes']==0 and not status['downloadRequired']
    monkeypatch.setattr(manager,'_digest',RuntimeSwitch._digest.__get__(manager))
    monkeypatch.setattr('core.torch_runtime_switch.urllib.request.urlopen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('network used')))
    state=dict(transactionId='a'*32,targetRuntime='cu126')
    RuntimeSwitch._download(manager,state,'torch',records['torch'])
    assert state['downloadCacheHits']==['torch']


def test_retained_rows_expose_safe_history_actions_and_pending_delete_is_not_reusable(manager,monkeypatch):
    first=manager.prepare('cu126');token=first['transactionId'];manager.keep_retained(token)
    status=manager.status()
    assert status['canDelete'] and status['retainedRuntimes'][0]['canDelete']
    manager.capability=(12,0)
    assert not manager.status()['retainedRuntimes'][0]['canDelete']
    manager.capability=(6,1)
    monkeypatch.setattr(manager,'_remove_tree',lambda *a:(_ for _ in ()).throw(PermissionError('busy before removal')))
    with pytest.raises(PermissionError):manager.delete_retained(token,True)
    row=manager.status()['retainedRuntimes'][0]
    assert row['deleting'] and not row['canReuse'] and row['canDelete']


def test_broken_windows_junction_is_refused_by_manager_and_startup_guard(manager,tmp_path):
    outside=tmp_path/'junction-target';outside.mkdir();manager._mkdir(manager.managed.parent)
    result=subprocess.run(['cmd.exe','/d','/c','mklink','/J',str(manager.managed),str(outside)],capture_output=True,text=True)
    if result.returncode:pytest.skip('Windows junction unavailable: '+result.stderr)
    outside.rmdir()
    try:
        assert manager.managed.is_junction() and not manager.managed.exists()
        with pytest.raises(ValueError,match='Linked'):manager.status()
        with pytest.raises(RuntimeError,match='路径异常'):assert_runtime_available(manager.root)
    finally:os.rmdir(manager.managed)


def test_hardware_probe_decodes_unreadable_stderr_conservatively(manager,monkeypatch):
    options=[]
    def run(command,**kwargs):
        options.append(kwargs);return subprocess.CompletedProcess(command,1,'','unreadable driver error')
    monkeypatch.setattr('core.torch_runtime_switch.subprocess.run',run)
    assert RuntimeSwitch._hardware(manager)==[]
    assert options[0]['encoding']=='utf-8' and options[0]['errors']=='replace'


def test_prefix_install_generates_record_paths_and_launchers_for_the_active_python(manager,tmp_path):
    """A tiny real wheel proves --prefix semantics without downloading Torch."""
    wheel_dir=manager.root/'wheel-fixture';wheel_dir.mkdir()
    wheel=wheel_dir/'cuda_prefix_layout_probe-0.0.1-py3-none-any.whl'
    info='cuda_prefix_layout_probe-0.0.1.dist-info'
    content={'cuda_prefix_layout_probe.py':b'def main():\n print("prefix probe")\n',
        info+'/METADATA':b'Metadata-Version: 2.1\nName: cuda-prefix-layout-probe\nVersion: 0.0.1\n',
        info+'/WHEEL':b'Wheel-Version: 1.0\nGenerator: isolated-qa\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        info+'/entry_points.txt':b'[console_scripts]\ncuda-prefix-probe = cuda_prefix_layout_probe:main\n'}
    rows=[]
    for path,data in content.items():rows.append([path,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='),str(len(data))])
    rows.append([info+'/RECORD','',''])
    import io
    stream=io.StringIO();csv.writer(stream).writerows(rows);content[info+'/RECORD']=stream.getvalue().encode()
    with zipfile.ZipFile(wheel,'w') as z:
        for path,data in content.items():z.writestr(path,data)
    prefix=manager.managed/'staged'/('a'*32)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    result=subprocess.run([sys.executable,'-I','-B','-m','pip','--isolated','install','--no-index','--find-links',str(wheel_dir),
        '--no-deps','--no-compile','--no-cache-dir','--ignore-installed','--prefix',str(prefix),'cuda-prefix-layout-probe==0.0.1'],
        capture_output=True,text=True,env=env,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    site=prefix/'Lib/site-packages';assert (site/info/'RECORD').is_file()
    from core.gpu_runtime import verify_runtime_records
    assert verify_runtime_records(manager.root,names=['cuda-prefix-layout-probe'],environment=prefix)['cuda-prefix-layout-probe']>0
    launcher=prefix/'Scripts/cuda-prefix-probe.exe';assert launcher.is_file()
    assert str(Path(sys.executable)).encode() in launcher.read_bytes()
