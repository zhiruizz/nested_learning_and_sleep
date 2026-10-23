"""Fail-closed public selector; historical private acceptance is not bundled."""
class AcceptanceUnavailable(RuntimeError):
    pass

def select_executor(requested='REF', *, scope=None):
    if requested not in ('REF', 'auto', 'CACHE', 'HOST', 'BOTH'):
        raise ValueError('unknown executor')
    if requested in ('REF', 'auto'):
        return {'executor': 'REF', 'reason': 'no transferable acceptance supplied'}
    raise AcceptanceUnavailable('No public acceptance for this exact scope; use REF. '
                                'The low-level engineering module requires independent revalidation.')
