import os
from typing import Any, Callable, List

# Check if Opik should be active based on credentials
OPIK_ENABLED = bool(os.getenv("OPIK_API_KEY") or os.getenv("OPIK_WORKSPACE") or os.getenv("OPIK_PROJECT_NAME"))

def get_opik_callbacks() -> List[Any]:
    """Returns the OpikTracer callback if Opik is enabled, otherwise an empty list."""
    if OPIK_ENABLED:
        try:
            from opik.integrations.langchain import OpikTracer
            return [OpikTracer()]
        except ImportError:
            return []
    return []

def safe_track(*args, **kwargs) -> Callable:
    """A wrapper around opik.track that gracefully degrades to a no-op if Opik is disabled."""
    if OPIK_ENABLED:
        try:
            from opik import track
            return track(*args, **kwargs)
        except ImportError:
            pass
            
    # If used as @safe_track (without parens), the first arg is the function
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
        
    # If used as @safe_track(...) (with parens), it should return a decorator
    def decorator(func: Callable) -> Callable:
        import functools
        @functools.wraps(func)
        def wrapper(*f_args, **f_kwargs):
            return func(*f_args, **f_kwargs)
        return wrapper
    return decorator
