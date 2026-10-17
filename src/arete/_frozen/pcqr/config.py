"""Reader for the deliberately flat frozen YAML subset; no added dependency."""
import ast

def load(path):
    def scalar(s):
        s=s.strip()
        if s.startswith('[') and s.endswith(']'):
            return [scalar(x) for x in s[1:-1].split(',')] if s[1:-1].strip() else []
        try: return ast.literal_eval(s)
        except (ValueError,SyntaxError): return s
    result={}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        key,value=line.split(':',1)
        if key.strip() in result: raise ValueError('duplicate config field')
        result[key.strip()]=scalar(value)
    return result
