import sys, json
sys.argv = [sys.argv[0]] + sys.argv[1:]
import fuzz
op, params = sys.argv[1], json.loads(sys.argv[2])
fuzz.OPS[op] = (lambda: dict(params),) + fuzz.OPS[op][1:]
sys.argv = [sys.argv[0], op, "1", "0"]
sys.exit(fuzz.main())
