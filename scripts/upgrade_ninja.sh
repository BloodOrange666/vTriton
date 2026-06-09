#!/bin/bash
set -e
cd /tmp
python3 -c "
import urllib.request, zipfile, os, sys
url = 'https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip'
print(f'Downloading {url}...')
urllib.request.urlretrieve(url, '/tmp/ninja-1.12.1.zip')
with zipfile.ZipFile('/tmp/ninja-1.12.1.zip', 'r') as z:
    z.extractall('/tmp/ninja-1.12.1')
ninja = '/tmp/ninja-1.12.1/ninja'
os.chmod(ninja, 0o755)
print(f'Extracted: {ninja}')
"
mkdir -p "$HOME/.local/bin"
cp /tmp/ninja-1.12.1/ninja "$HOME/.local/bin/ninja"
chmod +x "$HOME/.local/bin/ninja"
export PATH="$HOME/.local/bin:$PATH"
which ninja
ninja --version
echo "Ninja upgraded to: $(ninja --version) at $(which ninja)"
echo "NOTE: Ensure ~/.local/bin is in PATH for subsequent commands"
