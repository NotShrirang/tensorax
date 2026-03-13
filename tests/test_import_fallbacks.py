import builtins
import importlib
import sys


def _import_blocker_for_tensorax_c(original_import):
    def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'tensorax._C':
            raise ImportError('forced _C import failure')
        if name == 'tensorax' and fromlist and '_C' in fromlist:
            raise ImportError('forced _C import failure')
        return original_import(name, globals, locals, fromlist, level)

    return _blocked


def test_functional_module_exposes_backend_handle():
    import tensorax.functional as functional_module

    assert hasattr(functional_module, '_C')


def test_shape_utils_import_fallback_sets_none(monkeypatch):
    import tensorax
    import tensorax.utils.shape_utils as shape_utils_module

    original_import = builtins.__import__
    cached_c = sys.modules.pop('tensorax._C', None)
    had_attr = hasattr(tensorax, '_C')
    old_attr = getattr(tensorax, '_C', None)
    if had_attr:
        delattr(tensorax, '_C')

    monkeypatch.setattr(builtins, '__import__', _import_blocker_for_tensorax_c(original_import))

    importlib.reload(shape_utils_module)
    assert shape_utils_module._C is None

    monkeypatch.setattr(builtins, '__import__', original_import)
    if cached_c is not None:
        sys.modules['tensorax._C'] = cached_c
    if had_attr:
        setattr(tensorax, '_C', old_attr)
    importlib.reload(shape_utils_module)


def test_tensorax_cuda_is_available_returns_bool():
    import tensorax

    assert isinstance(tensorax.cuda_is_available(), bool)
