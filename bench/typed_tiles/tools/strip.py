import re, sys, hashlib
t = open(sys.argv[1]).read()
t = '\n'.join(l for l in t.split('\n') if not l.startswith(('plugin ', 'import ')))
t = re.sub(r'(?<![.\w])[A-Za-z_]\w*(?![.\w])', '', t)
print(hashlib.md5(t.encode()).hexdigest()[:12])
