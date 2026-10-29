"""Private, standard-library-only Python environment transaction for the launcher."""
import argparse
import base64
import configparser
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import uuid
import zipfile

TOKEN=re.compile(r'^[0-9a-f]{32}$')
PHASES={'preparing','moving-old','creating','installing','verifying','committed','restoring','rolled-back'}
BOOTSTRAP={'pyvenv.cfg','.gitignore','runtime-install-owner.json'}|{'Scripts/'+name for name in
    ['python.exe','pythonw.exe','activate','activate.bat','Activate.ps1','deactivate.bat']}


class Installer:
    def __init__(self,root,lock_token,profile='production',cuda='auto',offline=False,wheelhouse=None,preserve_config=False):
        self.root=Path(os.path.abspath(root));self.checked(self.root)
        if not (self.root/'_internal/DeepFaceLab/me.py').is_file():raise ValueError('Not a complete PT source project')
        self.cache=self.checked(self.root/'.launcher-install/source');self.active=self.root/'.venv'
        self.journal=self.cache/'python-repair.json';self.base=self.checked(self.root/'_internal/python_base/python.exe')
        self.tool=self.checked(self.root/'tools/torch-runtime.py')
        self.profile,self.cuda,self.offline,self.preserve_config=profile,cuda,offline,preserve_config
        self.wheelhouse=Path(os.path.abspath(wheelhouse)) if wheelhouse else None
        self.lock_token=lock_token

    def checked(self,path):
        path=Path(os.path.abspath(path))
        if path!=self.root and not path.is_relative_to(self.root):raise ValueError('Installer path escapes the project')
        for parent in [path,*path.parents]:
            try:info=parent.lstat()
            except FileNotFoundError:info=None
            if info and (stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400):
                raise ValueError('Linked installer/runtime paths are not supported: '+str(parent))
            if parent==self.root:break
        return path

    def read(self,path):return json.loads(self.checked(path).read_text(encoding='utf-8-sig'))

    def write(self,path,item):
        path=self.checked(path);self.checked(path.parent).mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temporary.open('x',encoding='utf-8',newline='\n') as output:
                json.dump(item,output,ensure_ascii=False,separators=(',',':'));output.flush();os.fsync(output.fileno())
            self.checked(path);os.replace(temporary,path)
        finally:
            if temporary.exists():self.checked(temporary).unlink()

    def paths(self,state):
        token=state.get('transactionId','')
        if state.get('schemaVersion')!=1 or not TOKEN.fullmatch(token) or state.get('phase') not in PHASES:
            raise ValueError('Invalid Python repair journal; recovery evidence was preserved')
        backup=self.cache/('venv-backup-'+token);session=self.cache/('python-session-'+token)
        if state.get('backup')!=backup.relative_to(self.root).as_posix() or state.get('session')!=session.relative_to(self.root).as_posix():
            raise ValueError('Python repair paths do not match their transaction')
        if type(state.get('hadVenv')) is not bool:raise ValueError('Invalid original environment state')
        inventory=state.get('oldInventory',[])
        if not isinstance(inventory,list):raise ValueError('Invalid original environment inventory')
        seen=set()
        for row in inventory:
            self.relative(row.get('path'))
            if row['path'] in seen or type(row.get('size')) is not int or row['size']<0 or type(row.get('mtime_ns')) is not int or row['mtime_ns']<0:
                raise ValueError('Invalid original environment inventory entry')
            seen.add(row['path'])
        if state['phase']=='committed' and state.get('validated') is not True:raise ValueError('Committed repair lacks validation proof')
        if state.get('requirements'):
            path=self.checked(self.root.joinpath(*self.relative(state['requirements'])))
            if not path.is_relative_to(session/'locks'):raise ValueError('Repair requirements escaped their session')
        return self.checked(backup),self.checked(session)

    def save(self,state):self.paths(state);self.write(self.journal,state)

    def files(self,prefix):
        prefix=self.checked(prefix);rows=[];stack=[prefix]
        while stack:
            directory=stack.pop();self.checked(directory)
            for path in directory.iterdir():
                self.checked(path);info=path.lstat()
                if stat.S_ISDIR(info.st_mode):stack.append(path)
                elif stat.S_ISREG(info.st_mode):rows.append(dict(path=path.relative_to(prefix).as_posix(),size=info.st_size,mtime_ns=info.st_mtime_ns))
                else:raise ValueError('Unsupported installer file type')
        return sorted(rows,key=lambda row:row['path'])

    def remove(self,prefix,allowed):
        if not self.checked(prefix).exists():return
        rows=self.files(prefix)
        if any(row['path'] not in allowed for row in rows):raise ValueError('Unknown installer files were preserved')
        directories=[];stack=[prefix]
        while stack:
            directory=stack.pop();directories.append(directory)
            for path in directory.iterdir():
                self.checked(path)
                if path.is_dir():stack.append(path)
                elif prefix==self.active and path==prefix/'runtime-install-owner.json':continue
                else:path.unlink()
        for path in sorted(directories,key=lambda path:-len(path.parts)):
            if path==prefix:continue
            self.checked(path).rmdir()
        marker=prefix/'runtime-install-owner.json'
        if prefix==self.active and marker.exists():self.checked(marker).unlink()
        self.checked(prefix).rmdir()

    def owned_remove(self,prefix,inventory):
        expected={row['path']:row for row in inventory}
        if self.checked(prefix).exists() and any(expected.get(row['path'])!=row for row in self.files(prefix)):
            raise ValueError('Installer-owned files changed; recovery evidence was preserved')
        self.remove(prefix,set(expected))

    def command(self,args,state,log,check=True):
        _,session=self.paths(state);temporary=self.checked(session/'tmp');temporary.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',NN_PROCESS_COUNT='0',TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary),
                 CUDA_CACHE_PATH=str(temporary/'cuda-cache'),TORCH_EXTENSIONS_DIR=str(temporary/'torch-extensions'),XDG_CACHE_HOME=str(temporary/'cache'))
        result=subprocess.run([str(arg) for arg in args],cwd=self.root,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=1800,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.checked(session/log).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.stdout:print(result.stdout,end='',flush=True)
        if check and result.returncode:raise RuntimeError(str(log)+' failed: '+(result.stderr or result.stdout)[-1800:])
        return result

    def pip(self,python,args,state,log,check=True):
        code=("import ensurepip,pathlib,runpy,sys; w=pathlib.Path(ensurepip.__file__).parent/'_bundled'/('pip-'+ensurepip.version()+'-py3-none-any.whl'); "
              "assert w.is_file(),'Bundled pip is missing';sys.path.insert(0,str(w));sys.argv=['pip','--isolated']+sys.argv[1:];runpy.run_module('pip',run_name='__main__')")
        return self.command([python,'-I','-B','-c',code,*args],state,log,check)

    def validate(self,state,requirements=None,check=True):
        args=[self.active/'Scripts/python.exe','-I','-B',self.tool,'validate','--cuda-runtime',state.get('cudaRuntime',self.cuda),
              '--maintenance-transaction',state['transactionId']]
        if requirements:args+=['--requirements',requirements]
        result=self.command(args,state,'validate.log',check=False)
        if result.returncode==0:result=self.pip(self.active/'Scripts/python.exe',['check'],state,'pip-check.log',check=False)
        if check and result.returncode:raise RuntimeError('The project Python / locked dependency validation failed: '+(result.stderr or result.stdout)[-1800:])
        return result.returncode==0

    def preflight(self,sbom):
        site=self.checked(self.active/'Lib/site-packages');selected=set(sbom['profiles'][self.profile]);owned=set(BOOTSTRAP)
        seen=set()
        for distribution in metadata.distributions(path=[str(site)]):
            name=re.sub(r'[-_.]+','-',distribution.metadata.get('Name','')).lower()
            if not name or name in seen and name not in {'pip','setuptools','wheel'}:
                raise ValueError('Duplicate or invalid existing distribution; the environment was preserved')
            seen.add(name)
            if name not in selected|{'pip','setuptools','wheel'}:
                raise ValueError('Existing package outside selected profile was preserved: '+name+'. Select its matching profile before repair.')
            if name in {'torch','torchvision'} and distribution.version!=sbom['packages'][name]['version']:
                raise ValueError('Existing CUDA variant must be changed through the launcher CUDA switch, not dependency repair')
            if not distribution.files:raise ValueError('Existing package lacks ownership RECORD: '+name)
            for file in distribution.files:
                path=self.checked(distribution.locate_file(file))
                if not path.is_relative_to(self.active):raise ValueError('Existing RECORD escapes its environment')
                owned.add(path.relative_to(self.active).as_posix())
        for row in self.files(self.active):
            relative=row['path'];path=PurePosixPath(relative)
            generated=(path.parent.name=='__pycache__' and path.name.endswith('.pyc') and
                       (path.parent.parent/(path.name.split('.cpython-')[0]+'.py')).as_posix() in owned)
            if relative not in owned and not generated:raise ValueError('Unknown existing environment file was preserved: '+relative)

    def bootstrap_duplicates(self):
        counts={}
        for distribution in metadata.distributions(path=[str(self.checked(self.active/'Lib/site-packages'))]):
            name=re.sub(r'[-_.]+','-',distribution.metadata.get('Name','')).lower()
            if name in {'pip','setuptools','wheel'}:counts[name]=counts.get(name,0)+1
        return sorted(name for name,count in counts.items() if count>1)

    @staticmethod
    def relative(value):
        if (not isinstance(value,str) or not value or ':' in value or '\\' in value or PurePosixPath(value).is_absolute()
                or any(part in {'','.','..'} for part in value.split('/'))):raise ValueError('Unsafe wheel file path')
        return PurePosixPath(value).parts

    def new_paths(self,wheels,state):
        allowed=set(BOOTSTRAP)
        for name,record,path in wheels:
            with zipfile.ZipFile(path) as archive:
                info=name.replace('-','_')+'-'+record['version']+'.dist-info'
                for member in archive.infolist():
                    if member.is_dir():continue
                    parts=self.relative(member.filename)
                    if stat.S_ISLNK(member.external_attr>>16):raise ValueError('Linked wheel member')
                    if parts[0].endswith('.data'):
                        if len(parts)<3 or parts[1] not in {'purelib','platlib','scripts','data'}:raise ValueError('Unsupported wheel data scheme')
                        prefix={'purelib':'Lib/site-packages/','platlib':'Lib/site-packages/','scripts':'Scripts/','data':''}[parts[1]]
                        relative=prefix+'/'.join(parts[2:])
                    else:relative='Lib/site-packages/'+member.filename
                    self.relative(relative);allowed.add(relative)
                for member in archive.namelist():
                    if member.endswith('.dist-info/entry_points.txt'):
                        config=configparser.ConfigParser(interpolation=None);config.optionxform=str
                        if archive.getinfo(member).file_size>1024*1024:raise ValueError('Oversized wheel entry points')
                        config.read_string(archive.read(member).decode('utf-8'))
                        for group in ['console_scripts','gui_scripts']:
                            for script in config.options(group) if config.has_section(group) else []:
                                if not re.fullmatch(r'[A-Za-z0-9_.-]+',script):raise ValueError('Unsafe wheel console script')
                                allowed.update('Scripts/'+script+suffix for suffix in ['', '.exe','-script.py'])
                    if member.endswith('.dist-info/METADATA'):info=member.split('/')[0]
                allowed.update('Lib/site-packages/'+info+'/'+item for item in ['RECORD','INSTALLER','REQUESTED'])
        _,session=self.paths(state);self.write(session/'new-ownership.json',dict(transactionId=state['transactionId'],files=sorted(allowed)))
        return allowed

    def cached_wheels(self,sbom,state):
        _,session=self.paths(state);directory=self.checked(self.cache/'wheels');directory.mkdir(parents=True,exist_ok=True)
        selected=sorted(sbom['profiles'][self.profile]);found=[];missing=[]
        for name in selected:
            record=sbom['packages'][name];wheel=record['wheel'];self.relative(wheel['file'])
            if len(PurePosixPath(wheel['file']).parts)!=1:raise ValueError('Invalid pinned wheel filename')
            candidates=([self.wheelhouse/wheel['file']] if self.wheelhouse else [])+[directory/wheel['file']]
            path=None
            for candidate in candidates:
                if candidate.parent==directory:self.checked(candidate)
                if not candidate.is_file():continue
                with candidate.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
                if candidate.stat().st_size==wheel['sizeBytes'] and digest==wheel['sha256']:path=candidate;break
                if candidate.parent!=directory:raise ValueError('Offline wheel hash mismatch; read-only input was preserved: '+str(candidate))
                self.checked(candidate).unlink()  # Only the managed pinned cache slot is replaceable.
            if path:found.append((name,record,path))
            else:missing.append((name,record))
        if missing:
            if self.offline:raise ValueError('Offline wheelhouse/cache is incomplete; the existing environment was preserved')
            lock=session/'missing.txt';lock.write_text('\n'.join(f"{record['name']}=={record['version']} --hash=sha256:{record['wheel']['sha256']}" for _,record in missing)+'\n',encoding='utf-8')
            self.pip(self.base,['download','--disable-pip-version-check','--no-cache-dir','--only-binary=:all:','--no-deps','--require-hashes',
                '--dest',directory,'--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/'+state['cudaRuntime'],'-r',lock],state,'download.log')
            for name,record in missing:
                wheel=record['wheel'];path=self.checked(directory/wheel['file'])
                with path.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
                if path.stat().st_size!=wheel['sizeBytes'] or digest!=wheel['sha256']:
                    path.unlink();raise ValueError('Downloaded pinned wheel failed SHA/size verification: '+name)
                found.append((name,record,path))
        return found

    def cleanup_session(self,state):
        _,session=self.paths(state)
        if not session.exists():return
        # This fixed transaction directory contains only generated plans/logs/tmp.
        self.remove(session,{row['path'] for row in self.files(session)})

    def restore_configuration(self,state):
        if not state.get('configurationChanged'):return
        _,session=self.paths(state);receipt=self.read(session/'original-configuration.json')
        original=base64.b64decode(receipt['original'],validate=True)
        if (receipt.get('schemaVersion')!=1 or receipt.get('transactionId')!=state['transactionId'] or receipt.get('originalSha256')!=hashlib.sha256(original).hexdigest()
                or not re.fullmatch(r'[0-9a-f]{64}',receipt.get('rewrittenSha256',''))):raise ValueError('Invalid original Python configuration receipt')
        path=self.checked(self.active/'pyvenv.cfg');current=hashlib.sha256(path.read_bytes()).hexdigest()
        if current==receipt['originalSha256']:return
        if current!=receipt['rewrittenSha256']:raise ValueError('Python configuration was edited during repair; unknown changes were preserved')
        temporary=self.checked(session/'configuration.restore.tmp');temporary.write_bytes(original)
        os.replace(temporary,path)

    def recover(self,state,validate_committed=True):
        backup,session=self.paths(state)
        if state['phase']=='committed':
            if not (self.active/'Scripts/python.exe').is_file() or not state.get('validated'):raise ValueError('Committed repair lacks its validated active environment')
            if backup.exists() and validate_committed:
                owner=self.read(self.cache/'source-install-owner.json')
                state.update(lockToken=self.lock_token,lockOwnerPid=owner['pid']);self.save(state)
                requirements=self.root/state['requirements'] if state.get('requirements') else None
                if requirements is not None and not self.checked(requirements).is_file():raise ValueError('Committed repair lock is missing')
                self.validate(state,requirements)
            self.owned_remove(backup,state.get('oldInventory',[]))
        elif backup.exists() or not state.get('hadVenv'):
            if backup.exists() and self.files(backup)!=state.get('oldInventory'):
                raise ValueError('Original environment changed; both recovery sides were preserved')
            state['phase']='restoring';self.save(state)
            if self.active.exists():
                marker=self.active/'runtime-install-owner.json'
                if not marker.exists() and self.files(self.active):raise ValueError('New environment owner marker is missing; unknown files were preserved')
                if marker.exists() and self.read(marker).get('transactionId')!=state['transactionId']:raise ValueError('New environment owner differs; recovery was refused')
                plan=self.read(session/'new-ownership.json') if (session/'new-ownership.json').exists() else dict(files=sorted(BOOTSTRAP))
                if 'transactionId' in plan and plan['transactionId']!=state['transactionId']:raise ValueError('Invalid new environment ownership')
                allowed=set(plan['files'])
                for value in allowed:self.relative(value)
                self.remove(self.active,allowed)
            if backup.exists():
                backup.rename(self.active)
            if state['hadVenv']:self.restore_configuration(state)
            state['phase']='rolled-back';self.save(state)
        elif state['phase']=='restoring' and state['hadVenv'] and self.active.exists() and self.files(self.active)==state.get('oldInventory'):
            self.restore_configuration(state);state['phase']='rolled-back';self.save(state)
        elif state['phase'] not in {'preparing','moving-old','rolled-back'}:
            raise ValueError('Original environment backup is missing; no unknown environment was changed')
        elif state['phase']!='rolled-back':
            self.restore_configuration(state)
            state['phase']='rolled-back';self.save(state)
        marker=self.active/'runtime-install-owner.json'
        if marker.exists():
            if self.read(marker).get('transactionId')!=state['transactionId']:raise ValueError('Unknown environment owner marker')
            marker.unlink()
        self.cleanup_session(state);self.checked(self.journal).unlink()

    def run(self,install=False):
        self.cache.mkdir(parents=True,exist_ok=True)
        owner=self.read(self.cache/'source-install-owner.json')
        if (not TOKEN.fullmatch(self.lock_token or '') or owner.get('token')!=self.lock_token or owner.get('schemaVersion')!=1):
            raise ValueError('Installer must hold the launcher installation lock')
        sys.path.insert(0,str(self.root/'_internal/DeepFaceLab'))
        from core.torch_runtime_switch import _existing_lock_busy,_pid_alive
        if not _pid_alive(owner.get('pid')) or not _existing_lock_busy(self.cache/'source-install.lock'):
            raise ValueError('Launcher installation lock is not active')
        if self.journal.exists():self.recover(self.read(self.journal))
        token=uuid.uuid4().hex
        state=dict(schemaVersion=1,transactionId=token,phase='preparing',hadVenv=self.active.exists(),lockToken=self.lock_token,
            lockOwnerPid=owner['pid'],backup=(self.cache/('venv-backup-'+token)).relative_to(self.root).as_posix(),
            session=(self.cache/('python-session-'+token)).relative_to(self.root).as_posix(),cudaRuntime=self.cuda)
        _,session=self.paths(state);session.mkdir(parents=True);self.save(state)
        success=False
        try:
            if state['hadVenv'] and not self.preserve_config:
                configuration=self.checked(self.active/'pyvenv.cfg')
                if configuration.is_file():
                    previous=configuration.read_bytes();value=previous.decode('utf-8-sig')
                    replacements={'home':str(self.base.parent),'executable':str(self.base),'command':str(self.base)+' -m venv '+str(self.active)}
                    for key,replacement in replacements.items():value=re.sub(r'(?m)^'+key+r'\s*=.*$',lambda match:key+' = '+replacement,value)
                    rewritten=value.encode('utf-8')
                    if rewritten!=previous:
                        self.write(session/'original-configuration.json',dict(schemaVersion=1,transactionId=token,original=base64.b64encode(previous).decode(),
                            originalSha256=hashlib.sha256(previous).hexdigest(),rewrittenSha256=hashlib.sha256(rewritten).hexdigest()))
                        state['configurationChanged']=True;self.save(state)
                        temporary=session/'configuration.next.tmp';temporary.write_bytes(rewritten);os.replace(temporary,configuration)
            requirements=None;sbom=None
            if install or self.profile!='production':
                result=self.command([self.base,'-I','-B',self.tool,'resolve','--cuda-runtime',self.cuda,'--profile',self.profile,'--output-dir',session/'locks'],state,'resolve.log')
                plan=json.loads(result.stdout);state['cudaRuntime']=plan['cudaRuntime']
                if state['cudaRuntime'] not in {'cu126','cu128'} or plan['indexUrl']!='https://download.pytorch.org/whl/'+state['cudaRuntime']:raise ValueError('Unexpected resolved CUDA index')
                requirements=self.checked(plan['requirements']);sbom_path=self.checked(plan['sbom'])
                if not requirements.is_relative_to(session/'locks') or not sbom_path.is_relative_to(session/'locks'):raise ValueError('Resolved locks escaped their session')
                sbom=self.read(sbom_path);state['requirements']=requirements.relative_to(self.root).as_posix();self.save(state)
            if ((self.active/'Scripts/python.exe').is_file() and self.validate(state,requirements,check=False)
                    and not self.bootstrap_duplicates()):
                state.update(phase='committed',validated=True);self.save(state);success=True
                self.recover(state,validate_committed=False);print('[runtime-install] Existing environment validated; no package installation needed.');return
            self.restore_configuration(state)
            if not install:raise ValueError('Existing environment needs explicit dependency repair; no packages were changed')
            if state['hadVenv']:self.preflight(sbom)
            wheels=self.cached_wheels(sbom,state);self.new_paths(wheels,state)
            backup,_=self.paths(state)
            if state['hadVenv']:
                state['oldInventory']=self.files(self.active);state['phase']='moving-old';self.save(state);self.active.rename(backup)
            state['phase']='creating';self.save(state);self.active.mkdir()
            self.write(self.active/'runtime-install-owner.json',dict(transactionId=token))
            self.command([self.base,'-I','-B','-m','venv','--without-pip',self.active],state,'venv.log')
            state['phase']='installing';self.save(state)
            find_links=sorted({str(path.parent) for _,_,path in wheels});args=['install','--disable-pip-version-check','--no-cache-dir','--no-compile','--no-deps','--only-binary=:all:','--no-index','--require-hashes','-r',requirements]
            for directory in find_links:args+=['--find-links',directory]
            self.pip(self.active/'Scripts/python.exe',args,state,'install.log')
            state['phase']='verifying';self.save(state);self.validate(state,requirements)
            state.update(phase='committed',validated=True)
            try:self.save(state)
            except BaseException:
                state['phase']='verifying';raise
            success=True
            self.recover(state,validate_committed=False);print('[runtime-install] Locked environment installed and validated.')
        except BaseException:
            if not success:
                try:self.recover(state)
                except BaseException as recovery:raise RuntimeError('Python setup failed; recovery must be retried before starting tasks: '+str(recovery))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True);parser.add_argument('--source-lock-token',required=True)
    parser.add_argument('--install-dependencies',action='store_true');parser.add_argument('--no-network',action='store_true')
    parser.add_argument('--wheelhouse');parser.add_argument('--profile',default='production');parser.add_argument('--cuda-runtime',default='auto',choices=['auto','cu126','cu128'])
    parser.add_argument('--preserve-configuration',action='store_true');args=parser.parse_args()
    Installer(args.project_root,args.source_lock_token,args.profile,args.cuda_runtime,args.no_network,args.wheelhouse,args.preserve_configuration).run(args.install_dependencies)


if __name__=='__main__':main()
