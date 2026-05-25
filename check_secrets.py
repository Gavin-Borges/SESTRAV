import os, re
pattern = re.compile(r'(?i)(api_key|token|secret|password|passwd|auth)\s*=\s*[\'\"].{4,}[\'\"]')
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith(('.py', '.sh', '.ps1', '.yaml')):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    for i, line in enumerate(lines):
                        if pattern.search(line):
                            print(f'{p}:{i+1}: {line.strip()}')
            except Exception:  # nosec B110
                pass
