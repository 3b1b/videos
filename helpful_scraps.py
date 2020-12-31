# Terminal namespace issue
pairs = list(locals().items())
for name, obj in pairs:
    if isinstance(obj, Mobject):
        globals()[name] = obj
