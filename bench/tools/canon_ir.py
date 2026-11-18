# canon_ir.py <file.ll>: prints the LLVM IR with gid suffixes and metadata ids renumbered by first appearance, so two
# builds' device IR compare equal when only their numbering differs.
import re, sys

text = open(sys.argv[1]).read()
ids = {}


def renumber(match):
    return '_' + ids.setdefault(match.group(1), str(len(ids)))


text = re.sub(r'_(\d{4,})(?!\d)', renumber, text)
sys.stdout.write(re.sub(r'!\d+', '!N', text))
