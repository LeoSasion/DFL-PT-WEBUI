"""Exercise the real Windows installers with isolated native runtime fixtures."""
import importlib.util
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows runtime installer integration')

FAKE_RUNTIME = r'''
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Diagnostics;
using System.IO.Compression;
using System.Web.Script.Serialization;
class FakeRuntime {
    static string Value(string[] args, string flag) {
        int i=Array.IndexOf(args,flag); return i>=0 && i+1<args.Length ? args[i+1] : null;
    }
    static bool Has(string[] args,string value) { return Array.IndexOf(args,value)>=0; }
    static int Main(string[] args) {
        string exe=Assembly.GetExecutingAssembly().Location;
        var json=new JavaScriptSerializer();
        string log=Environment.GetEnvironmentVariable("DFL_TEST_CALL_LOG");
        string req=Value(args,"-r");
        File.AppendAllText(log,json.Serialize(new {exe=exe,args=args,
            temp=Environment.GetEnvironmentVariable("TEMP"),tmp=Environment.GetEnvironmentVariable("TMP"),
            noBytecode=Environment.GetEnvironmentVariable("PYTHONDONTWRITEBYTECODE"),
            requirements=req!=null && File.Exists(req) ? File.ReadAllText(req) : null})+"\n");
        string name=Path.GetFileName(exe).ToLowerInvariant();
        if(name=="node.exe") { Console.WriteLine("24.19.0"); return 0; }
        if(name=="ffmpeg.exe" || name=="ffprobe.exe") { Console.WriteLine("ffmpeg version 9.0.1"); return 0; }
        if(Has(args,"--source-lock-token")) {
            var start=new ProcessStartInfo(Environment.GetEnvironmentVariable("DFL_TEST_HELPER_PYTHON"));
            start.UseShellExecute=false;
            var quoted=new List<string>();
            foreach(string arg in args) quoted.Add("\""+arg.Replace("\\\"","\\\\\\\"").Replace("\"","\\\"")+"\"");
            start.Arguments=String.Join(" ",quoted.ToArray());
            var child=Process.Start(start); child.WaitForExit(); return child.ExitCode;
        }
        if(Has(args,"venv")) {
            string target=args[args.Length-1];
            Directory.CreateDirectory(Path.Combine(target,"Scripts"));
            File.Copy(exe,Path.Combine(target,"Scripts","python.exe"));
            File.WriteAllText(Path.Combine(target,"pyvenv.cfg"),"home = fixture\nversion = 3.12.14\n");
            return 0;
        }
        string selected=Environment.GetEnvironmentVariable("DFL_TEST_SELECTED_CUDA") ?? "cu128";
        string project=Environment.GetEnvironmentVariable("DFL_TEST_PROJECT");
        if(Has(args,"resolve")) {
            string output=Value(args,"--output-dir"); Directory.CreateDirectory(output);
            string path=Path.Combine(output,"resolved.txt");
            string fixturePlan=Path.Combine(project,"fixture-plans",selected+".json");
            var plan=json.Deserialize<Dictionary<string,object>>(File.ReadAllText(fixturePlan));
            File.WriteAllText(path,(string)plan["lock"]);
            string sbom=Path.Combine(output,"resolved.json"); File.WriteAllText(sbom,json.Serialize(plan["sbom"]));
            string index=Environment.GetEnvironmentVariable("DFL_TEST_BAD_INDEX") ?? "https://download.pytorch.org/whl/"+selected;
            Console.WriteLine(json.Serialize(new {cudaRuntime=selected,indexUrl=index,requirements=path,sbom=sbom}));
            return 0;
        }
        if(Has(args,"download")) {
            if(Environment.GetEnvironmentVariable("DFL_TEST_FAIL")=="download") return 8;
            string dest=Value(args,"--dest"); Directory.CreateDirectory(dest);
            foreach(string source in Directory.GetFiles(Path.Combine(project,"fixture-wheels",selected),"*.whl")) File.Copy(source,Path.Combine(dest,Path.GetFileName(source)),true);
            return 0;
        }
        if(Has(args,"install")) {
            string site=Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(exe)),"Lib","site-packages");Directory.CreateDirectory(site);
            foreach(string source in Directory.GetFiles(Value(args,"--find-links"),"*.whl")) ZipFile.ExtractToDirectory(source,site);
            return Environment.GetEnvironmentVariable("DFL_TEST_FAIL")=="install" ? 9 : 0;
        }
        if(Has(args,"check") && Environment.GetEnvironmentVariable("DFL_TEST_FAIL")=="check") return 7;
        if(Has(args,"validate")) {
            string oldMarker=Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(exe)),"Lib","site-packages","torch","broken-marker");
            if(File.Exists(oldMarker) || Environment.GetEnvironmentVariable("DFL_TEST_FAIL")=="validate") return 6;
            string expected=Value(args,"--cuda-runtime");
            string installed=Environment.GetEnvironmentVariable("DFL_TEST_INSTALLED_CUDA");
            if(!String.IsNullOrEmpty(installed) && expected!="auto" && expected!=installed) {
                Console.Error.WriteLine("Existing runtime does not match requested CUDA profile"); return 3;
            }
            string actual=String.IsNullOrEmpty(installed) ? selected : installed;
            Console.WriteLine("DFL_PT_RUNTIME_OK|3.12.14|2.9.1+"+actual+"|12."+(actual=="cu126" ? "6" : "8"));
            return 0;
        }
        Console.WriteLine("Python base: 3.12.14"); return 0;
    }
}
'''


@pytest.fixture(scope='module')
def fake_exe(tmp_path_factory):
    directory = tmp_path_factory.mktemp('fake-windows-runtimes')
    source, target = directory / 'FakeRuntime.cs', directory / 'FakeRuntime.exe'
    source.write_text(FAKE_RUNTIME, encoding='utf-8')
    compiler = Path(os.environ['WINDIR']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    assert compiler.is_file()
    result = subprocess.run([str(compiler), '/nologo', '/target:exe', '/r:System.Web.Extensions.dll','/r:System.IO.Compression.FileSystem.dll',
                             '/out:' + str(target), str(source)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return target


def fixture(tmp_path, fake_exe, *, installed=None, selected='cu126',broken=False):
    project = tmp_path / 'project'
    for rel in ['_internal/python_base/python.exe', '_internal/node/bin/node.exe',
                '_internal/ffmpeg/ffmpeg.exe', '_internal/ffmpeg/ffprobe.exe']:
        p = project / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fake_exe, p)
    for rel in ['_internal/DeepFaceLab/me.py', 'requirements.txt', 'webui/pnpm-lock.yaml',
                'webui/pnpm-workspace.yaml', 'tools/prepare-vision-runtime.ps1', 'tools/torch-runtime.py',
                '_internal/node/bin/npm.cmd', '_internal/node/bin/node_modules/corepack/dist/corepack.js']:
        p = project / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# fixture', encoding='utf-8')
    shutil.copy2(ROOT/'tools/runtime-install.py',project/'tools/runtime-install.py')
    core=project/'_internal/DeepFaceLab/core';core.mkdir()
    for name in ['gpu_runtime.py','torch_runtime_switch.py','download_sources.py']:
        shutil.copy2(ROOT/'_internal/DeepFaceLab/core'/name,core/name)
    (core/'__init__.py').write_text('',encoding='utf-8')
    for variant in ['cu126','cu128']:
        directory=project/'fixture-wheels'/variant;directory.mkdir(parents=True)
        version='2.9.1+'+variant;info='torch-'+version+'.dist-info'
        content={'torch/__init__.py':b'synthetic package',info+'/METADATA':('Metadata-Version: 2.1\nName: torch\nVersion: '+version+'\n').encode()}
        stream=io.StringIO();rows=[]
        for name,data in content.items():rows.append([name,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='),str(len(data))])
        rows.append([info+'/RECORD','','']);csv.writer(stream).writerows(rows);content[info+'/RECORD']=stream.getvalue().encode()
        wheel=directory/('torch-'+version+'-py3-none-any.whl')
        with zipfile.ZipFile(wheel,'w') as archive:
            for name,data in content.items():archive.writestr(name,data)
        digest=hashlib.sha256(wheel.read_bytes()).hexdigest()
        plan={'lock':f'torch=={version} --hash=sha256:{digest}\n','sbom':{'profiles':{'production':['torch']},'packages':{
            'torch':{'name':'torch','version':version,'wheel':{'file':wheel.name,'sizeBytes':wheel.stat().st_size,'sha256':digest}}}}}
        plans=project/'fixture-plans';plans.mkdir(exist_ok=True);(plans/(variant+'.json')).write_text(json.dumps(plan))
    for rel, data in [('release/version.json', {'nodeVersion': '24.19.0', 'pnpmVersion': '11.19.0'}),
                      ('webui/package.json', {'packageManager': 'pnpm@11.19.0'})]:
        p = project / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding='utf-8')
    for name in ['torch-runtimes.json','python-sbom.json','download-sources.json']:shutil.copy2(ROOT/'release'/name,project/'release'/name)
    if installed:
        p = project / '.venv/Scripts/python.exe'
        p.parent.mkdir(parents=True)
        shutil.copy2(fake_exe, p)
        (project / '.venv/pyvenv.cfg').write_text('home = unchanged-fixture\nversion = 3.12.14\n', encoding='utf-8')
        site=project/'.venv/Lib/site-packages';site.mkdir(parents=True)
        with zipfile.ZipFile(next((project/'fixture-wheels'/installed).glob('*.whl'))) as archive:archive.extractall(site)
        if broken:
            marker=site/'torch/broken-marker';marker.write_bytes(b'broken fixture package')
            with (site/('torch-2.9.1+'+installed+'.dist-info/RECORD')).open('a',encoding='utf-8',newline='') as stream:csv.writer(stream).writerow(['torch/broken-marker','',''])
    protected = project / 'workspace/model/tae.pt'
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b'existing fixture checkpoint must not change')
    log = tmp_path / 'runtime-calls.jsonl'
    env = dict(os.environ, DFL_TEST_CALL_LOG=str(log), DFL_TEST_SELECTED_CUDA=selected,
               DFL_TEST_HELPER_PYTHON=sys.executable,DFL_TEST_PROJECT=str(project),
               TEMP=str(tmp_path), TMP=str(tmp_path), TMPDIR=str(tmp_path))
    env.pop('DFL_TEST_BAD_INDEX', None)
    env.pop('DFL_TEST_FAIL',None)
    if installed:
        env['DFL_TEST_INSTALLED_CUDA'] = installed
    else:
        env.pop('DFL_TEST_INSTALLED_CUDA', None)
    return project, log, env


def run_script(name, project, log, env, *options):
    command = ['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
               '-File', str(ROOT / 'launcher' / name), '-ProjectRoot', str(project), *map(str, options)]
    result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True,encoding='utf-8',errors='replace', timeout=60)
    calls = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()] if log.exists() else []
    assert (project / 'workspace/model/tae.pt').read_bytes() == b'existing fixture checkpoint must not change'
    return result, calls


def installation(calls):
    return [row for row in calls if 'install' in row['args'] and any('pip' in arg for arg in row['args'])]


@pytest.mark.parametrize('variant', ['cu126', 'cu128'])
@pytest.mark.parametrize('offline', [False, True])
def test_source_install_uses_resolved_hash_lock_and_matching_index(tmp_path, fake_exe, variant, offline):
    project, log, env = fixture(tmp_path, fake_exe, selected=variant)
    options = ['-SkipVisionAssets', '-SkipWebuiPreparation', '-CudaRuntime', variant]
    if offline:
        wheelhouse = project / 'fixture-wheels'/variant
        options += ['-NoNetwork', '-WheelhousePath', wheelhouse]
    result, calls = run_script('install-source.ps1', project, log, env, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    install, = installation(calls)
    assert '--require-hashes' in install['args']
    assert f'torch==2.9.1+{variant} --hash=sha256:' in install['requirements']
    assert '--no-index' in install['args'] and '--extra-index-url' not in install['args']
    downloads=[row for row in calls if 'download' in row['args']]
    if offline:assert not downloads
    else:assert downloads[0]['args'][downloads[0]['args'].index('--extra-index-url')+1] == f'https://download.pytorch.org/whl/{variant}'
    validation = next(row for row in calls if 'validate' in row['args'])
    assert validation['args'][validation['args'].index('--cuda-runtime')+1] == variant
    assert '--requirements' in validation['args']
    assert not list((project / '.launcher-install/source').glob('installing-*'))


@pytest.mark.parametrize('mismatch', [False, True])
def test_source_install_preserves_existing_environment(tmp_path, fake_exe, mismatch):
    project, log, env = fixture(tmp_path, fake_exe, installed='cu128' if mismatch else 'cu126')
    before = (project / '.venv/pyvenv.cfg').read_bytes()
    result, calls = run_script('install-source.ps1', project, log, env,
                              '-SkipVisionAssets', '-SkipWebuiPreparation', '-CudaRuntime', 'cu126')
    assert (result.returncode != 0) == mismatch, result.stdout + result.stderr
    assert not installation(calls) and not any('venv' in row['args'] for row in calls)
    assert (project / '.venv/pyvenv.cfg').read_bytes() == before


def test_setup_startup_accepts_existing_legacy_runtime_without_installing(tmp_path, fake_exe):
    project, log, env = fixture(tmp_path, fake_exe, installed='cu126')
    result, calls = run_script('setup-runtime.ps1', project, log, env, '-NoNetwork')
    assert result.returncode == 0, result.stdout + result.stderr
    assert not installation(calls) and not any('resolve' in row['args'] for row in calls)
    validation, = [row for row in calls if 'validate' in row['args']]
    assert validation['args'][validation['args'].index('--cuda-runtime')+1] == 'auto'


@pytest.mark.parametrize('offline', [False, True])
def test_setup_explicit_install_reuses_profile_lock_and_cleans_selection(tmp_path, fake_exe, offline):
    project, log, env = fixture(tmp_path, fake_exe, installed='cu126',broken=True)
    options = ['-InstallDependencies', '-CudaRuntime', 'cu126']
    if offline:
        wheelhouse = project / 'fixture-wheels/cu126'
        options += ['-NoNetwork', '-WheelhousePath', wheelhouse]
    result, calls = run_script('setup-runtime.ps1', project, log, env, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    install, = installation(calls)
    assert 'torch==2.9.1+cu126 --hash=sha256:' in install['requirements']
    assert '--require-hashes' in install['args']
    assert '--no-index' in install['args']
    if not offline:
        download=next(row for row in calls if 'download' in row['args'])
        assert download['args'][download['args'].index('--extra-index-url')+1].endswith('/cu126')
    assert not list((project / '.launcher-install').glob('runtime-locks-*'))


@pytest.mark.parametrize('script', ['install-source.ps1', 'setup-runtime.ps1'])
def test_unexpected_resolver_index_is_rejected_before_install(tmp_path, fake_exe, script):
    project, log, env = fixture(tmp_path, fake_exe, installed='cu126')
    env['DFL_TEST_BAD_INDEX'] = 'https://example.invalid/wheels'
    options = ['-SkipVisionAssets', '-SkipWebuiPreparation','-RepairDependencies'] if script.startswith('install-') else ['-InstallDependencies']
    result, calls = run_script(script, project, log, env, *options)
    assert result.returncode != 0 and not installation(calls)


def environment_bytes(project):
    return {str(path.relative_to(project/'.venv')):path.read_bytes() for path in (project/'.venv').rglob('*') if path.is_file()}


def test_explicit_repair_validates_healthy_environment_without_pip_or_cfg_changes(tmp_path,fake_exe):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126');before=environment_bytes(project)
    result,calls=run_script('install-source.ps1',project,log,env,'-RepairDependencies','-SkipVisionAssets','-SkipWebuiPreparation')
    assert result.returncode==0,result.stdout+result.stderr
    assert not installation(calls) and not any('download' in row['args'] or 'venv' in row['args'] for row in calls)
    assert environment_bytes(project)==before
    assert not (project/'.launcher-install/source/python-repair.json').exists()


@pytest.mark.parametrize('script',['install-source.ps1','setup-runtime.ps1'])
@pytest.mark.parametrize('missing',['runtime-tool','private-api'])
def test_legacy_source_is_refused_before_runtime_downloads_or_environment_changes(tmp_path,fake_exe,script,missing):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126');before=environment_bytes(project)
    if missing=='runtime-tool':
        (project/'tools/torch-runtime.py').unlink()
        (project/'_internal/DeepFaceLab/core/gpu_runtime.py').unlink()
    else:
        (project/'_internal/DeepFaceLab/core/gpu_runtime.py').write_text('def validate_runtime(root):\n pass\n')
    options=['-RepairDependencies','-SkipVisionAssets','-SkipWebuiPreparation'] if script.startswith('install-') else ['-InstallDependencies']
    result,calls=run_script(script,project,log,env,*options)
    assert result.returncode!=0 and 'predatestransactionalruntimerepair' in re.sub(r'\s+','',result.stdout+result.stderr)
    assert not calls and environment_bytes(project)==before
    assert not (project/'.launcher-install/source/python-repair.json').exists()


@pytest.mark.parametrize('failure',['download','install','validate','check'])
def test_failed_existing_repair_restores_bytes_and_reuses_verified_cache_on_retry(tmp_path,fake_exe,failure):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126',broken=True);before=environment_bytes(project)
    env['DFL_TEST_FAIL']=failure
    result,calls=run_script('setup-runtime.ps1',project,log,env,'-InstallDependencies')
    assert result.returncode!=0 and environment_bytes(project)==before,result.stdout+result.stderr
    assert not (project/'.launcher-install/source/python-repair.json').exists()
    assert not list((project/'.launcher-install/source').glob('venv-backup-*'))
    downloads=sum('download' in row['args'] for row in calls)
    env.pop('DFL_TEST_FAIL')
    result,calls=run_script('setup-runtime.ps1',project,log,env,'-InstallDependencies')
    assert result.returncode==0,result.stdout+result.stderr
    assert sum('download' in row['args'] for row in calls)==downloads+(failure=='download')
    assert not list((project/'.launcher-install/source').glob('python-session-*'))
    assert not (project/'.venv/Lib/site-packages/torch/broken-marker').exists()
    actions=[row for row in calls if 'install' in row['args'] or 'download' in row['args']]
    assert all(str(project/'.launcher-install/source/python-session-').lower() in row['temp'].lower() for row in actions)


@pytest.mark.parametrize('kind',['extra-package','unknown-file','cuda-mismatch'])
def test_existing_optional_packages_unknown_files_and_other_cuda_are_preserved(tmp_path,fake_exe,kind):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu128' if kind=='cuda-mismatch' else 'cu126',broken=True)
    if kind=='unknown-file':(project/'.venv/custom-user.cfg').write_bytes(b'user configuration')
    if kind=='extra-package':
        extra=project/'.venv/Lib/site-packages/optional_tool-1.0.dist-info';extra.mkdir()
        (extra/'METADATA').write_text('Metadata-Version: 2.1\nName: optional-tool\nVersion: 1.0\n')
        (extra/'RECORD').write_text('optional_tool-1.0.dist-info/METADATA,,\noptional_tool-1.0.dist-info/RECORD,,\n')
    before=environment_bytes(project)
    result,calls=run_script('setup-runtime.ps1',project,log,env,'-InstallDependencies','-CudaRuntime','cu126')
    assert result.returncode!=0 and environment_bytes(project)==before
    assert not installation(calls) and not any('download' in row['args'] for row in calls)


def test_commit_journal_failure_rolls_back_before_backup_cleanup(tmp_path,fake_exe):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126',broken=True);before=environment_bytes(project)
    helper=project/'tools/runtime-install.py';source=helper.read_text()
    assert 'try:self.save(state)\n            except BaseException:' in source
    helper.write_text(source.replace('try:self.save(state)\n            except BaseException:',
        'try:raise PermissionError("injected durable commit failure")\n            except BaseException:'))
    result,_=run_script('setup-runtime.ps1',project,log,env,'-InstallDependencies')
    assert result.returncode!=0 and environment_bytes(project)==before
    assert not (project/'.launcher-install/source/python-repair.json').exists()


def add_duplicate_bootstrap_metadata(project,name='pip'):
    infos=[]
    for version in ['0.1','0.2']:
        info=project/'.venv/Lib/site-packages'/(name+'-'+version+'.dist-info');info.mkdir(parents=True)
        (info/'METADATA').write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\n')
        (info/'RECORD').write_text(info.name+'/METADATA,,\n'+info.name+'/RECORD,,\n')
        infos.append(info)
    return infos


@pytest.mark.parametrize('failure',[False,True])
def test_duplicate_pip_metadata_is_not_healthy_and_explicit_repair_is_reversible(tmp_path,fake_exe,failure):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126');add_duplicate_bootstrap_metadata(project);before=environment_bytes(project)
    result,calls=run_script('setup-runtime.ps1',project,log,env,'-NoNetwork')
    assert result.returncode!=0 and environment_bytes(project)==before and not installation(calls)
    assert any('check' in row['args'] for row in calls)
    if failure:env['DFL_TEST_FAIL']='install'
    result,calls=run_script('setup-runtime.ps1',project,log,env,'-InstallDependencies')
    if failure:assert result.returncode!=0 and environment_bytes(project)==before
    else:
        assert result.returncode==0,result.stdout+result.stderr
        assert not list((project/'.venv/Lib/site-packages').glob('pip-*.dist-info'))
    assert not (project/'.launcher-install/source/python-repair.json').exists()


def test_committed_cleanup_failure_never_rolls_back_and_next_setup_retries(tmp_path,fake_exe):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126',broken=True)
    helper=project/'tools/runtime-install.py';source=helper.read_text()
    helper.write_text(source.replace("            self.owned_remove(backup,state.get('oldInventory',[]))",
        "            if (self.root/'block-cleanup').exists():raise PermissionError('injected backup cleanup failure')\n            self.owned_remove(backup,state.get('oldInventory',[]))"))
    blocker=project/'block-cleanup';blocker.write_bytes(b'busy')
    result,_=run_script('setup-runtime.ps1',project,log,env,'-InstallDependencies')
    assert result.returncode!=0
    state=json.loads((project/'.launcher-install/source/python-repair.json').read_text())
    assert state['phase']=='committed' and state['validated']
    assert not (project/'.venv/Lib/site-packages/torch/broken-marker').exists()
    blocker.unlink()
    result,calls=run_script('setup-runtime.ps1',project,log,env,'-NoNetwork')
    assert result.returncode==0,result.stdout+result.stderr
    assert len(installation(calls))==1 and not (project/'.launcher-install/source/python-repair.json').exists()
    assert not list((project/'.launcher-install/source').glob('venv-backup-*'))


@pytest.mark.parametrize('unknown_edit',[False,True])
def test_hard_exit_after_rebase_recovers_original_configuration_without_overwriting_edits(tmp_path,fake_exe,unknown_edit):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126',broken=True)
    configuration=project/'.venv/pyvenv.cfg';before=configuration.read_bytes()
    helper=project/'tools/runtime-install.py';source=helper.read_text()
    needle='temporary.write_bytes(rewritten);os.replace(temporary,configuration)'
    assert needle in source;helper.write_text(source.replace(needle,needle+';os._exit(73)'))
    result,_=run_script('setup-runtime.ps1',project,log,env,'-NoNetwork')
    assert result.returncode!=0 and configuration.read_bytes()!=before
    assert (project/'.launcher-install/source/python-repair.json').exists()
    helper.write_text(source)
    if unknown_edit:configuration.write_bytes(b'home = manually-edited\n')
    result,_=run_script('setup-runtime.ps1',project,log,env,'-NoNetwork')
    assert result.returncode!=0
    if unknown_edit:
        assert configuration.read_bytes()==b'home = manually-edited\n'
        assert (project/'.launcher-install/source/python-repair.json').exists()
    else:
        assert configuration.read_bytes()==before
        assert not (project/'.launcher-install/source/python-repair.json').exists()


@pytest.mark.parametrize('obstruction',['none','unknown-active','changed-backup'])
def test_interrupted_repair_recovers_before_new_validation_and_preserves_unknown_files(tmp_path,fake_exe,obstruction):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126',broken=True);before=environment_bytes(project)
    spec=importlib.util.spec_from_file_location('isolated_runtime_install',ROOT/'tools/runtime-install.py')
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    manager=tool.Installer(project,'a'*32);token='b'*32;cache=project/'.launcher-install/source';cache.mkdir(parents=True)
    backup=cache/('venv-backup-'+token);session=cache/('python-session-'+token);session.mkdir()
    state={'schemaVersion':1,'transactionId':token,'phase':'installing','hadVenv':True,'oldInventory':manager.files(project/'.venv'),
        'backup':backup.relative_to(project).as_posix(),'session':session.relative_to(project).as_posix()}
    (project/'.venv').rename(backup);active=project/'.venv';(active/'Lib/site-packages/torch').mkdir(parents=True)
    (active/'runtime-install-owner.json').write_text(json.dumps({'transactionId':token}))
    (active/'Lib/site-packages/torch/__init__.py').write_bytes(b'partial new install')
    (session/'new-ownership.json').write_text(json.dumps({'transactionId':token,'files':['runtime-install-owner.json','Lib/site-packages/torch/__init__.py']}))
    if obstruction=='unknown-active':(active/'do-not-delete.txt').write_bytes(b'user-added')
    if obstruction=='changed-backup':(backup/'pyvenv.cfg').write_bytes(b'edited backup configuration')
    (cache/'python-repair.json').write_text(json.dumps(state))
    result,_=run_script('setup-runtime.ps1',project,log,env,'-NoNetwork')
    assert result.returncode!=0  # Recovered original still needs explicit package repair.
    if obstruction=='unknown-active':
        assert (active/'do-not-delete.txt').read_bytes()==b'user-added' and backup.exists()
        assert (cache/'python-repair.json').exists()
    elif obstruction=='changed-backup':
        assert (active/'Lib/site-packages/torch/__init__.py').read_bytes()==b'partial new install'
        assert (active/'runtime-install-owner.json').exists()
        assert (backup/'pyvenv.cfg').read_bytes()==b'edited backup configuration'
    else:
        assert environment_bytes(project)==before and not backup.exists()
        assert not (cache/'python-repair.json').exists()
        result,_=run_script('setup-runtime.ps1',project,log,env,'-NoNetwork')
        assert result.returncode!=0 and environment_bytes(project)==before


def test_interrupted_partial_rollback_keeps_owner_marker_and_can_resume(tmp_path,fake_exe):
    project,log,env=fixture(tmp_path,fake_exe,installed='cu126',broken=True);before=environment_bytes(project)
    helper=project/'tools/runtime-install.py';source=helper.read_text()
    needle='                else:path.unlink()'
    assert needle in source
    replacement="""                else:
                    if prefix==self.active:
                        self._qa_unlinks=getattr(self,'_qa_unlinks',0)+1
                        if self._qa_unlinks==2:raise PermissionError('second rollback unlink blocked')
                    path.unlink()"""
    helper.write_text(source.replace(needle,replacement));env['DFL_TEST_FAIL']='install'
    result,_=run_script('setup-runtime.ps1',project,log,env,'-InstallDependencies')
    assert result.returncode!=0
    assert (project/'.venv/runtime-install-owner.json').exists()
    state=json.loads((project/'.launcher-install/source/python-repair.json').read_text())
    assert state['phase']=='restoring'
    helper.write_text(source);env.pop('DFL_TEST_FAIL')
    result,_=run_script('setup-runtime.ps1',project,log,env,'-NoNetwork')
    assert result.returncode!=0 and environment_bytes(project)==before
    assert not (project/'.launcher-install/source/python-repair.json').exists()


def test_archive_cache_reuses_verification_and_repairs_only_managed_invalid_files(tmp_path):
    project=tmp_path/'project';cache=project/'.launcher-install/source';archives=cache/'archives';archives.mkdir(parents=True)
    source=tmp_path/'tiny.zip'
    with zipfile.ZipFile(source,'w') as archive:archive.writestr('runtime/readme.txt','tiny official fixture')
    digest=hashlib.sha256(source.read_bytes()).hexdigest();cached=archives/'runtime.zip';cached.write_bytes(b'corrupt cache')
    def ps(path):return str(path).replace("'","''")
    script=r"""
$ErrorActionPreference='Stop';$root='ROOT';$rootPrefix=$root+'\';$cache='CACHE';$ArchiveDirectory='ARCHIVES';$NoNetwork=$false
$source='SOURCE';$DownloadCount=0;$HashCount=0
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Utility/Microsoft.PowerShell.Utility.psd1') -Force
$ast=[Management.Automation.Language.Parser]::ParseFile('INSTALLER',[ref]$null,[ref]$null)
foreach($name in @('Assert-ProjectPath','Get-VerifiedArchive')) {
  Invoke-Expression ($ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$true)[0].Extent.Text)
}
    function Invoke-DflPinnedDownload { param($Url,$OutFile,$Sha256) $script:DownloadCount++;Copy-Item -LiteralPath $source -Destination $OutFile }
function Get-FileHash {
  param($LiteralPath,$Algorithm) $script:HashCount++
  $inputFile=[IO.File]::OpenRead($LiteralPath);$digest=[Security.Cryptography.SHA256]::Create()
  try { @{Hash=[BitConverter]::ToString($digest.ComputeHash($inputFile)).Replace('-','')} }
  finally { $inputFile.Dispose();$digest.Dispose() }
}
$one=Get-VerifiedArchive 'runtime.zip' 'https://example.invalid/runtime.zip' 'DIGEST'
$two=Get-VerifiedArchive 'runtime.zip' 'https://example.invalid/runtime.zip' 'DIGEST'
if($DownloadCount -ne 1 -or $HashCount -ne 3 -or $one -ne $two){throw 'Cache verification/download was repeated'}
@{downloads=$DownloadCount;hashes=$HashCount;path=$two}|ConvertTo-Json -Compress
"""
    for key,value in [('ROOT',project),('CACHE',cache),('ARCHIVES',archives),('SOURCE',source),('INSTALLER',ROOT/'launcher/install-source.ps1')]:script=script.replace(key,ps(value))
    script=script.replace('DIGEST',digest)
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert cached.read_bytes()==source.read_bytes() and not list(cache.glob('*.partial'))


def test_real_without_pip_venv_install_matches_the_predeclared_cleanup_inventory(tmp_path):
    project=tmp_path/'project';(project/'_internal/DeepFaceLab').mkdir(parents=True)
    (project/'_internal/DeepFaceLab/me.py').write_text('# isolated layout fixture')
    spec=importlib.util.spec_from_file_location('runtime_install_layout_probe',ROOT/'tools/runtime-install.py')
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    manager=tool.Installer(project,'a'*32);token='b'*32
    state=dict(schemaVersion=1,transactionId=token,phase='creating',hadVenv=False,
        backup='.launcher-install/source/venv-backup-'+token,session='.launcher-install/source/python-session-'+token)
    session=project/state['session'];session.mkdir(parents=True)
    directory=tmp_path/'wheels';directory.mkdir();name='micro_runtime_install';info=name+'-0.0.1.dist-info'
    files={name+'.py':b'def main():\n print("micro installer ready")\n',info+'/METADATA':b'Metadata-Version: 2.1\nName: micro-runtime-install\nVersion: 0.0.1\n',
        info+'/WHEEL':b'Wheel-Version: 1.0\nGenerator: isolated-qa\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        info+'/entry_points.txt':b'[console_scripts]\nmicro-runtime-install = micro_runtime_install:main\n'}
    rows=[[path,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='),str(len(data))] for path,data in files.items()]
    rows.append([info+'/RECORD','','']);stream=io.StringIO();csv.writer(stream).writerows(rows);files[info+'/RECORD']=stream.getvalue().encode()
    wheel=directory/(name+'-0.0.1-py3-none-any.whl')
    with zipfile.ZipFile(wheel,'w') as archive:
        for path,data in files.items():archive.writestr(path,data)
    requirements=session/'requirements.txt';requirements.write_text('micro-runtime-install==0.0.1 --hash=sha256:'+hashlib.sha256(wheel.read_bytes()).hexdigest()+'\n')
    allowed=manager.new_paths([('micro-runtime-install',{'version':'0.0.1'},wheel)],state)
    manager.command([ROOT/'_internal/python_base/python.exe','-I','-B','-m','venv','--without-pip',manager.active],state,'real-venv.log')
    manager.pip(manager.active/'Scripts/python.exe',['install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','-r',requirements,'--find-links',directory],state,'real-install.log')
    assert all(row['path'] in allowed for row in manager.files(manager.active))
    assert not (manager.active/'Lib/site-packages/pip').exists()
    result=manager.command([manager.active/'Scripts/python.exe','-I','-B','-c','import micro_runtime_install;micro_runtime_install.main()'],state,'real-import.log')
    assert 'micro installer ready' in result.stdout
    assert str(manager.active/'Scripts/python.exe').encode() in (manager.active/'Scripts/micro-runtime-install.exe').read_bytes()
    duplicate_infos=add_duplicate_bootstrap_metadata(project)
    assert manager.pip(manager.active/'Scripts/python.exe',['check'],state,'real-duplicate-pip-check.log').returncode==0
    assert manager.bootstrap_duplicates()==['pip']
    for info in duplicate_infos:manager.remove(info,{'METADATA','RECORD'})
    manager.remove(manager.active,allowed);assert not manager.active.exists()


def test_manifest_accepts_paired_variants_and_rejects_crossed_cuda_versions():
    manifest = json.loads((ROOT / 'launcher/runtime-manifest.json').read_text(encoding='utf-8'))
    python = next(c for c in manifest['components'] if c['id'] == 'python')
    pattern = python['validation']['command']['outputRegex']
    assert manifest['manifestVersion'] == '2026.10.30.1'
    for variant, cuda in [('cu126','12.6'), ('cu128','12.8')]:
        assert re.fullmatch(pattern, f'DFL_PT_RUNTIME_OK|3.12.14|2.9.1+{variant}|{cuda}')
    assert not re.fullmatch(pattern, 'DFL_PT_RUNTIME_OK|3.12.14|2.9.1+cu126|12.8')
    assert not re.fullmatch(pattern, 'DFL_PT_RUNTIME_OK|3.12.14|2.9.1+cu128|12.6')


def test_smoke_runtime_uses_shared_validator_and_reports_its_error(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('smoke_runtime_integration', ROOT / 'tools/smoke_test.py')
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    tools = tmp_path / 'tools'
    tools.mkdir()
    (tools / 'torch-runtime.py').write_text("def validate_runtime(root):\n    raise ValueError('incompatible-wheel-fixture')\n", encoding='utf-8')
    (tools / 'runtime-dependencies.py').write_text('def direct(path):\n    return []\n', encoding='utf-8')
    messages = []
    monkeypatch.setattr(smoke, 'ROOT', tmp_path)
    monkeypatch.setattr(smoke, 'failed', messages.append)
    monkeypatch.setattr(smoke, 'passed', lambda message: None)
    monkeypatch.setattr(smoke, 'run_check', lambda *args, **kwargs: None)
    smoke.check_runtime()
    assert any('incompatible-wheel-fixture' in message for message in messages)
