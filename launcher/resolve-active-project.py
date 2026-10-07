"""Read the same active-project registry as WebUI; never creates or selects one.

Only this small resolver runs before legacy BAT tools. A missing registry means
the default project; malformed or unregistered IDs fail closed.
"""
import argparse
import json
from pathlib import Path
import re
import sys


def resolve(root):
    root = Path(root).resolve()
    registry = root / 'webui/.runtime/projects.json'
    if not registry.exists():
        project = {'id': 'default', 'name': '默认项目'}
    else:
        record = json.loads(registry.read_text(encoding='utf-8-sig'))
        active = record.get('activeId')
        if not isinstance(active, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', active):
            raise ValueError('Invalid active project ID; reopen WebUI project selection')
        matches = [item for item in record.get('projects', []) if item.get('id') == active]
        if len(matches) != 1:
            raise ValueError('Active project is not registered; reopen WebUI project selection')
        project = matches[0]
    workspace = root / ('workspace' if project['id'] == 'default' else 'workspaces/' + project['id'])
    if workspace.is_symlink() or (hasattr(workspace, 'is_junction') and workspace.is_junction()) or not workspace.resolve().is_relative_to(root):
        raise ValueError('Active workspace escapes the project')
    return {'id': project['id'], 'name': project.get('name', project['id']), 'workspace': str(workspace)}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--field', choices=('workspace', 'id', 'bat', 'json'), default='json')
    args = parser.parse_args()
    try:
        value = resolve(args.root)
        print(json.dumps(value, ensure_ascii=False) if args.field == 'json' else
              value['id'] + '|' + value['workspace'] if args.field == 'bat' else value[args.field])
    except Exception as error:
        print('Active-project resolution failed: ' + str(error), file=sys.stderr)
        sys.exit(2)
