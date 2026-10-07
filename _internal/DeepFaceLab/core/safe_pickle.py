"""Bounded, data-only reader for historical DFL pickle files.

This is deliberately an opcode interpreter, not an Unpickler with a GLOBAL
allowlist. No callable from the input is imported or invoked. NumPy recipes
are interpreted as data and their size is checked before allocating arrays.
The writer remains pickle-compatible with existing facesets and checkpoints.
"""
from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
from pathlib import Path, PureWindowsPath, PurePosixPath
import pickletools
import re
import shutil
import struct

import numpy as np


class DataFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    bytes: int
    nodes: int
    array_bytes: int
    total_array_bytes: int
    depth: int = 48
    rank: int = 8


PROFILES = {
    'metadata': Limits(65533, 100000, 1024 * 1024, 2 * 1024 * 1024),
    'faceset': Limits(128 * 1024 * 1024, 4000000, 16 * 1024 * 1024, 128 * 1024 * 1024),
    'session': Limits(128 * 1024 * 1024, 4000000, 16 * 1024 * 1024, 128 * 1024 * 1024),
    'weights': Limits(1024 * 1024 * 1024, 2000000, 512 * 1024 * 1024, 1024 * 1024 * 1024),
}


@dataclass(frozen=True)
class _Symbol:
    module: str
    name: str


@dataclass
class _DType:
    code: str
    endian: str = '='

    def value(self):
        # Numeric/bool only: object, strings, structs and subarrays have no
        # supported executable or variable-size reconstruction path here.
        if not re.fullmatch(r'[?biufc][124816]?', self.code):
            raise DataFormatError('Unsupported NumPy dtype')
        dtype = np.dtype(self.endian + self.code)
        if dtype.hasobject or dtype.fields or dtype.subdtype or dtype.itemsize > 16:
            raise DataFormatError('Unsupported NumPy dtype layout')
        return dtype


@dataclass
class _Array:
    value: object = None


@dataclass
class DataRecord:
    """Inert legacy session fields, never an instance of the input class."""
    type_name: str
    state: object = None


_SESSION_RECORDS = {
    ('merger.InteractiveMergerSubprocessor', 'InteractiveMergerSubprocessor.Frame'),
    ('merger.FrameInfo', 'FrameInfo'),
    ('merger.MergerConfig', 'MergerConfigMasked'),
}
_SESSION_PATHS = {('pathlib', name) for name in ('WindowsPath', 'PosixPath', 'PureWindowsPath', 'PurePosixPath')}


_NUMPY_MULTI = ('numpy.core.multiarray', 'numpy._core.multiarray')
_NUMPY_NUMERIC = ('numpy.core.numeric', 'numpy._core.numeric')
_ENUMS = {
    ('samplelib.Sample', 'SampleType'): {0, 1, 2, 3},
    ('facelib.FaceType', 'FaceType'): {0, 1, 2, 3, 4, 10, 20, 100},
    ('core.imagelib.SegIEPolys', 'SegIEPolyType'): {0, 1},
}


def _symbol(module, name, *, session=False):
    pair = (module, name)
    if (pair in _ENUMS or pair in {('numpy', 'dtype'), ('numpy', 'ndarray'),
                                  ('_codecs', 'encode'), ('collections', 'OrderedDict')}
            or module in _NUMPY_MULTI and name in ('_reconstruct', 'scalar')
            or module in _NUMPY_NUMERIC and name == '_frombuffer'
            or session and pair in _SESSION_RECORDS | _SESSION_PATHS):
        return _Symbol(module, name)
    raise DataFormatError(f'Forbidden pickle global: {module}.{name}')


class _Reader:
    def __init__(self, limits, profile):
        self.limits = limits
        self.profile = profile
        self.array_bytes = 0
        self.operations = 0
        self.materialized_nodes = 0

    def array(self, raw, dtype, shape, order):
        if not isinstance(dtype, _DType) or not isinstance(raw, (bytes, bytearray)):
            raise DataFormatError('Invalid NumPy array recipe')
        if not isinstance(shape, (tuple, list)) or len(shape) > self.limits.rank:
            raise DataFormatError('Invalid NumPy array rank')
        count = 1
        for dim in shape:
            if type(dim) is not int or dim < 0 or dim > 100000000:
                raise DataFormatError('Invalid NumPy array dimension')
            count *= dim
            if count > self.limits.array_bytes:
                raise DataFormatError('NumPy array element budget exceeded')
        dtype_value = dtype.value()
        size = count * dtype_value.itemsize
        if (size != len(raw) or size > self.limits.array_bytes
                or self.array_bytes + size > self.limits.total_array_bytes):
            raise DataFormatError('NumPy array byte budget or payload mismatch')
        if order not in ('C', 'F'):
            raise DataFormatError('Invalid NumPy array order')
        self.array_bytes += size
        # Nothing from pickle is executed. Only validated fixed-size raw bytes
        # are copied into a numeric array, after every allocation bound above.
        return np.frombuffer(raw, dtype=dtype_value).reshape(tuple(shape), order=order).copy(order=order)

    def reduce(self, symbol, args):
        if not isinstance(symbol, _Symbol) or not isinstance(args, tuple):
            raise DataFormatError('Pickle REDUCE must be a supported data recipe')
        pair = (symbol.module, symbol.name)
        if self.profile == 'session' and pair in _SESSION_PATHS:
            if len(args) > 1024 or any(type(part) is not str or len(part) > 4096 for part in args):
                raise DataFormatError('Invalid lexical session path')
            pure_path = PureWindowsPath if 'Windows' in symbol.name else PurePosixPath
            return str(pure_path(*args))
        if pair in _ENUMS:
            if len(args) != 1 or type(args[0]) is not int or args[0] not in _ENUMS[pair]:
                raise DataFormatError('Invalid historical enum value')
            return args[0]
        if pair == ('numpy', 'dtype'):
            if not 1 <= len(args) <= 3 or not isinstance(args[0], str) or len(args[0]) > 8:
                raise DataFormatError('Invalid NumPy dtype recipe')
            code = args[0]
            endian = code[0] if code[0] in '<>|=' else '='
            result = _DType(code[1:] if code[0] in '<>|=' else code, endian)
            result.value()
            return result
        if symbol.module in _NUMPY_MULTI and symbol.name == '_reconstruct':
            if args != (_Symbol('numpy', 'ndarray'), (0,), b'b'):
                raise DataFormatError('Unsupported NumPy reconstruction recipe')
            return _Array()
        if symbol.module in _NUMPY_NUMERIC and symbol.name == '_frombuffer':
            if len(args) != 4:
                raise DataFormatError('Invalid NumPy buffer recipe')
            return _Array(self.array(args[0], args[1], args[2], args[3]))
        if symbol.module in _NUMPY_MULTI and symbol.name == 'scalar':
            if len(args) != 2:
                raise DataFormatError('Invalid NumPy scalar recipe')
            return self.array(args[1], args[0], (), 'C').item()
        if pair == ('_codecs', 'encode'):
            if len(args) != 2 or args[1] != 'latin1' or not isinstance(args[0], str):
                raise DataFormatError('Only historical Latin-1 byte data is supported')
            return args[0].encode('latin1')
        if pair == ('collections', 'OrderedDict'):
            if not args:
                return {}
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return dict(args[0])
        raise DataFormatError('Unsupported pickle data recipe')

    def build(self, target, state):
        if isinstance(target, DataRecord) and target.state is None:
            if not isinstance(state, dict) or len(state) > 512 or any(type(key) is not str for key in state):
                raise DataFormatError('Invalid legacy session object fields')
            target.state = state
            return
        if isinstance(target, _DType):
            if (not isinstance(state, tuple) or len(state) != 8 or state[0] != 3
                    or state[1] not in ('<', '>', '|', '=') or state[2:5] != (None, None, None)
                    or state[5:] != (-1, -1, 0)):
                raise DataFormatError('Unsupported NumPy dtype state')
            target.endian = state[1]
            target.value()
            return
        if isinstance(target, _Array) and target.value is None:
            if (not isinstance(state, tuple) or len(state) != 5 or state[0] != 1
                    or type(state[3]) is not bool):
                raise DataFormatError('Invalid NumPy array state')
            target.value = self.array(state[4], state[2], state[1], 'F' if state[3] else 'C')
            return
        raise DataFormatError('Forbidden pickle BUILD target')

    def parse(self, payload, *, stream=None, require_eof=True):
        stack, memo = [], {}
        mark = object()
        stream = BytesIO(payload) if stream is None else stream

        def marked():
            try:
                index = len(stack) - 1 - stack[::-1].index(mark)
            except ValueError as error:
                raise DataFormatError('Missing pickle MARK') from error
            values = stack[index + 1:]
            del stack[index:]
            return values

        def put_items(target, values):
            if not isinstance(target, dict) or len(values) % 2:
                raise DataFormatError('Invalid pickle dictionary')
            for key, value in zip(values[::2], values[1::2]):
                if type(key) not in (str, int, float, bool, bytes, tuple):
                    raise DataFormatError('Unsupported pickle dictionary key')
                target[key] = value

        try:
            for opcode, arg, position in self.opcodes(stream):
                self.operations += 1
                if self.operations > self.limits.nodes * 4 or len(stack) + len(memo) > self.limits.nodes:
                    raise DataFormatError('Pickle object/operation budget exceeded')
                name = opcode.name
                if name == 'STOP':
                    if len(stack) != 1 or require_eof and stream.tell() != len(payload):
                        raise DataFormatError('Invalid pickle stack or trailing data')
                    return self.materialize(stack[0])
                if name in ('PROTO', 'FRAME'):
                    if name == 'PROTO' and arg > 5 or name == 'FRAME' and arg > len(payload) - stream.tell():
                        raise DataFormatError('Unsupported/truncated pickle framing')
                elif name == 'MARK': stack.append(mark)
                elif name == 'NONE': stack.append(None)
                elif name in ('NEWTRUE', 'NEWFALSE'): stack.append(name == 'NEWTRUE')
                elif name in ('INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT',
                              'STRING', 'UNICODE', 'BINUNICODE', 'BINUNICODE8', 'SHORT_BINUNICODE',
                              'BINSTRING', 'SHORT_BINSTRING', 'BINBYTES', 'BINBYTES8', 'SHORT_BINBYTES', 'BYTEARRAY8'):
                    if type(arg) is int and arg.bit_length() > 128:
                        raise DataFormatError('Integer magnitude exceeded')
                    stack.append(arg)
                elif name in ('EMPTY_LIST', 'EMPTY_TUPLE', 'EMPTY_DICT'):
                    stack.append([] if name == 'EMPTY_LIST' else () if name == 'EMPTY_TUPLE' else {})
                elif name in ('LIST', 'TUPLE', 'DICT'):
                    values = marked()
                    if name == 'DICT':
                        value = {}; put_items(value, values); stack.append(value)
                    else: stack.append(values if name == 'LIST' else tuple(values))
                elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                    count = int(name[-1]); values = stack[-count:]; del stack[-count:]; stack.append(tuple(values))
                elif name == 'APPEND':
                    value = stack.pop()
                    if not isinstance(stack[-1], list): raise DataFormatError('Invalid pickle list')
                    stack[-1].append(value)
                elif name == 'APPENDS':
                    values = marked()
                    if not isinstance(stack[-1], list): raise DataFormatError('Invalid pickle list')
                    stack[-1].extend(values)
                elif name == 'SETITEM':
                    value, key = stack.pop(), stack.pop(); put_items(stack[-1], [key, value])
                elif name == 'SETITEMS':
                    values = marked(); put_items(stack[-1], values)
                elif name in ('PUT', 'BINPUT', 'LONG_BINPUT', 'MEMOIZE'):
                    index = len(memo) if name == 'MEMOIZE' else arg
                    if type(index) is not int or not 0 <= index < self.limits.nodes or index in memo:
                        raise DataFormatError('Invalid pickle memo index')
                    memo[index] = stack[-1]
                elif name in ('GET', 'BINGET', 'LONG_BINGET'): stack.append(memo[arg])
                elif name == 'GLOBAL':
                    module, symbol_name = arg.split(' ', 1); stack.append(_symbol(module, symbol_name, session=self.profile == 'session'))
                elif name == 'STACK_GLOBAL':
                    symbol_name, module = stack.pop(), stack.pop()
                    if type(module) is not str or type(symbol_name) is not str:
                        raise DataFormatError('Invalid pickle global names')
                    stack.append(_symbol(module, symbol_name, session=self.profile == 'session'))
                elif name == 'NEWOBJ' and self.profile == 'session':
                    args, symbol = stack.pop(), stack.pop()
                    if not isinstance(symbol, _Symbol) or (symbol.module, symbol.name) not in _SESSION_RECORDS or args != ():
                        raise DataFormatError('Forbidden session object constructor')
                    stack.append(DataRecord(symbol.module + '.' + symbol.name))
                elif name == 'REDUCE':
                    args, symbol = stack.pop(), stack.pop(); stack.append(self.reduce(symbol, args))
                elif name == 'BUILD': self.build(stack[-2], stack.pop())
                elif name == 'POP': stack.pop()
                elif name == 'POP_MARK': marked()
                elif name == 'DUP': stack.append(stack[-1])
                else:
                    raise DataFormatError(f'Forbidden pickle opcode: {name}')
        except DataFormatError:
            raise
        except (ValueError, TypeError, IndexError, KeyError, OverflowError, UnicodeError, struct.error) as error:
            raise DataFormatError(f'Malformed pickle data: {error}') from error
        raise DataFormatError('Truncated pickle data')

    def opcodes(self, stream):
        # Guard integer argument lengths before pickletools turns raw bytes
        # into a Python bigint. genops by itself applies no such bound.
        while True:
            position = stream.tell()
            code = stream.read(1)
            if not code: raise DataFormatError('Truncated pickle data')
            opcode = pickletools.code2op.get(chr(code[0]))
            if opcode is None: raise DataFormatError('Unknown pickle opcode')
            argument_position = stream.tell()
            if opcode.name in ('LONG1', 'LONG4'):
                size_bytes = stream.read(1 if opcode.name == 'LONG1' else 4)
                if len(size_bytes) != (1 if opcode.name == 'LONG1' else 4):
                    raise DataFormatError('Truncated integer length')
                size = size_bytes[0] if opcode.name == 'LONG1' else struct.unpack('<i', size_bytes)[0]
                if not 0 <= size <= 16: raise DataFormatError('Integer magnitude exceeded')
            elif opcode.name in ('INT', 'LONG'):
                line = stream.readline(128)
                if not line.endswith(b'\n'): raise DataFormatError('Integer text size exceeded')
            stream.seek(argument_position)
            argument = opcode.arg.reader(stream) if opcode.arg is not None else None
            yield opcode, argument, position
            if opcode.name == 'STOP': return

    def materialize(self, root):
        active = set()
        def visit(value, depth):
            self.materialized_nodes += 1
            if self.materialized_nodes > self.limits.nodes or depth > self.limits.depth:
                raise DataFormatError('Data tree size/depth exceeded')
            if isinstance(value, _Array):
                if value.value is None: raise DataFormatError('Incomplete NumPy array state')
                return value.value
            if isinstance(value, DataRecord):
                if value.state is None: raise DataFormatError('Incomplete legacy session record')
                identity = id(value)
                if identity in active: raise DataFormatError('Cyclic legacy session record')
                active.add(identity)
                result = DataRecord(value.type_name, visit(value.state, depth + 1))
                active.remove(identity)
                return result
            if type(value) in (type(None), str, bytes, bool, int, float): return value
            if type(value) is bytearray: return bytes(value)
            if type(value) not in (list, tuple, dict): raise DataFormatError('Unsupported data value')
            identity = id(value)
            if identity in active: raise DataFormatError('Cyclic data is unsupported')
            active.add(identity)
            if isinstance(value, dict):
                result = {visit(k, depth + 1): visit(v, depth + 1) for k, v in value.items()}
            else:
                result = [visit(v, depth + 1) for v in value]
                if isinstance(value, tuple): result = tuple(result)
            active.remove(identity)
            return result

        return visit(root, 0)


def loads(payload, *, profile='metadata'):
    limits = PROFILES[profile]
    if not isinstance(payload, (bytes, bytearray)) or not 0 < len(payload) <= limits.bytes:
        raise DataFormatError('Pickle payload size exceeded or empty')
    return _Reader(limits, profile).parse(payload)


def load_file(filename, *, profile='metadata'):
    path = Path(filename)
    limit = PROFILES[profile].bytes
    if not 0 < path.stat().st_size <= limit:
        raise DataFormatError('Pickle file size exceeded or empty')
    with path.open('rb') as stream:
        payload = stream.read(limit + 1)
    return loads(payload, profile=profile)


def load_records_file(filename, *, profile='weights', max_records=4096, return_sha256=False):
    """Read historical concatenated pickle records with one shared budget.

    This supports count + one-tensor records, without invoking native pickle.
    Each record has a fresh memo, matching separate pickle.dump calls.
    """
    path = Path(filename)
    limits = PROFILES[profile]
    if not 0 < path.stat().st_size <= limits.bytes or not 0 < max_records <= 500000:
        raise DataFormatError('Pickle records file/count size exceeded')
    with path.open('rb') as stream: payload = stream.read(limits.bytes + 1)
    if len(payload) > limits.bytes: raise DataFormatError('Pickle payload size exceeded')
    reader = _Reader(limits, profile)
    stream = BytesIO(payload)
    values = []
    while stream.tell() < len(payload):
        if len(values) >= max_records: raise DataFormatError('Pickle record count exceeded')
        values.append(reader.parse(payload, stream=stream, require_eof=False))
    return (values, hashlib.sha256(payload).hexdigest()) if return_sha256 else values


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
        raise DataFormatError(f'Invalid {name} coordinates')


def validate_dfl_metadata(value):
    if not isinstance(value, dict) or len(value) > 512 or any(type(k) is not str for k in value):
        raise DataFormatError('DFL metadata must be a string-keyed dictionary')
    for key in ('landmarks', 'source_landmarks'):
        points = value.get(key)
        if points is not None:
            array = np.asarray(points)
            if array.ndim != 2 or not 1 <= array.shape[0] <= 4096:
                raise DataFormatError(f'Invalid {key} count')
            _numeric(points, (array.shape[0], 2), key)
    for key, shape in (('image_to_face_mat', (2, 3)), ('source_rect', (4,))):
        if value.get(key) is not None: _numeric(value[key], shape, key)
    for key in ('face_type', 'source_filename'):
        if value.get(key) is not None and (type(value[key]) is not str or len(value[key]) > 4096):
            raise DataFormatError(f'Invalid {key}')
    if value.get('eyebrows_expand_mod') is not None:
        number = value['eyebrows_expand_mod']
        if type(number) not in (int, float) or not math.isfinite(number):
            raise DataFormatError('Invalid eyebrow expansion')
    mask = value.get('xseg_mask')
    if mask is not None:
        array = np.asarray(list(mask) if isinstance(mask, bytes) else mask)
        if array.ndim > 2 or array.size > 65533 or array.dtype.kind not in 'biuf':
            raise DataFormatError('Invalid compressed XSeg mask')
        if not np.isfinite(array).all() or (array < 0).any() or (array > 255).any():
            raise DataFormatError('Invalid compressed XSeg byte values')
        if not np.equal(array, np.floor(array)).all():
            raise DataFormatError('Compressed XSeg values must be integer bytes')
        raw = array.astype(np.uint8).tobytes()
        if raw.startswith(b'\x89PNG\r\n\x1a\n'):
            if len(raw) < 24: raise DataFormatError('Truncated PNG mask header')
            width, height = struct.unpack('>II', raw[16:24])
            if not 0 < width <= 4096 or not 0 < height <= 4096 or width * height > 4194304:
                raise DataFormatError('XSeg mask canvas exceeded')
        elif raw.startswith(b'\xff\xd8'):
            # Pillow reads the JPEG header lazily; never ask a decoder to
            # allocate the advertised mask canvas before checking its size.
            from PIL import Image
            try:
                with Image.open(BytesIO(raw)) as image: width, height = image.size
            except Exception as error:
                raise DataFormatError('Invalid JPEG mask header') from error
            if not 0 < width <= 4096 or not 0 < height <= 4096 or width * height > 4194304:
                raise DataFormatError('XSeg mask canvas exceeded')
    polys = value.get('seg_ie_polys')
    if polys is not None:
        entries = polys.get('polys') if isinstance(polys, dict) else polys
        if not isinstance(entries, list) or len(entries) > 4096:
            raise DataFormatError('Invalid segmentation polygons')
        for entry in entries:
            kind, points = (entry.get('type'), entry.get('pts')) if isinstance(entry, dict) else entry
            if kind not in (0, 1): raise DataFormatError('Invalid segmentation polygon type')
            array = np.asarray(points)
            if array.ndim != 2 or array.shape[0] > 16384:
                raise DataFormatError('Invalid segmentation polygon count')
            _numeric(points, (array.shape[0], 2), 'segmentation polygon')
    return value


def validate_faceset_configs(value):
    if not isinstance(value, list) or not len(value) <= 500000:
        raise DataFormatError('Faceset metadata must contain a bounded sample list')
    allowed = {'sample_type', 'filename', 'face_type', 'shape', 'landmarks', 'seg_ie_polys',
               'xseg_mask', 'xseg_mask_compressed', 'eyebrows_expand_mod', 'source_filename',
               'person_name', 'pitch_yaw_roll'}
    for config in value:
        if not isinstance(config, dict) or set(config) - allowed:
            raise DataFormatError('Invalid faceset sample configuration')
        if type(config.get('filename')) is not str or len(config['filename']) > 4096:
            raise DataFormatError('Invalid faceset sample filename')
        validate_shape(config.get('shape'))
        metadata = {k: v for k, v in config.items() if k in ('landmarks', 'seg_ie_polys', 'source_filename', 'eyebrows_expand_mod')}
        metadata['xseg_mask'] = config.get('xseg_mask_compressed')
        validate_dfl_metadata(metadata)
        if config.get('sample_type') not in (None, 0, 1, 2, 3):
            raise DataFormatError('Invalid sample type')
        if config.get('face_type') not in (None, 0, 1, 2, 3, 4, 10, 20, 100):
            raise DataFormatError('Invalid sample face type')
    return value


def validate_shape(shape):
    if (not isinstance(shape, (tuple, list)) or len(shape) != 3
            or any(type(n) is not int for n in shape) or not 0 < shape[0] <= 16384
            or not 0 < shape[1] <= 16384 or shape[2] not in (1, 3, 4)
            or shape[0] * shape[1] > 67108864):
        raise DataFormatError('Image shape exceeds the supported canvas')
    return tuple(shape)


def validate_metadata_backup(value):
    if not isinstance(value, dict) or len(value) > 500000:
        raise DataFormatError('Invalid faceset metadata backup')
    for name, saved in value.items():
        if type(name) is not str or Path(name).name != name or any(c in name for c in '/\\:'):
            raise DataFormatError('Unsafe metadata backup filename')
        if not isinstance(saved, (tuple, list)) or len(saved) != 2:
            raise DataFormatError('Invalid metadata backup entry')
        validate_shape(saved[0]); validate_dfl_metadata(saved[1])
    return value


def validate_session_mapping(value):
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise DataFormatError('Session data must be a string-keyed dictionary')
    if value.get('iter') is not None and (type(value['iter']) is not int or not 0 <= value['iter'] <= 2**63 - 1):
        raise DataFormatError('Invalid model iteration')
    if value.get('options') is not None and not isinstance(value['options'], dict):
        raise DataFormatError('Invalid model options')
    if value.get('loss_history') is not None and not isinstance(value['loss_history'], list):
        raise DataFormatError('Invalid model loss history')
    return value


def preserve_rejected_file(filename):
    """Keep invalid optional settings before a later save can replace them."""
    path = Path(filename).absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise DataFormatError('Rejected settings must be an ordinary file')
    def digest(target):
        result = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): result.update(block)
        return result.hexdigest()
    original_sha = digest(path)
    archive = path.with_name(path.name + '.invalid-' + original_sha[:16])
    if not archive.exists():
        with path.open('rb') as source, archive.open('xb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    if archive.is_symlink() or digest(archive) != original_sha or digest(path) != original_sha:
        raise DataFormatError('Cannot preserve rejected settings without losing original data')
    return archive
