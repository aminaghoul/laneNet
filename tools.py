from os.path import join

from sqlitedict import SqliteDict

cache = SqliteDict('/dev/shm/cache.sqlite3')


def cached_function(name):
    def wrapper(callback):
        # TODO: Args and kwargs
        def after_call(result):
            return result

        def wrapped():
            try:
                return wrapped.after_call(cache[name])
            except FileNotFoundError:
                cache[name] = result = callback()
                cache.commit()
                return wrapped.after_call(result)

        wrapped.after_call = after_call

        return wrapped

    return wrapper
