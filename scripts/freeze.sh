#!/usr/bin/env bash
pip freeze > requirements.txt

# strip Windows-only packages
sed -i '/^pywin32==/d;/^pywinpty==/d' requirements.txt

# strip CUDA wheels to CPU wheels
sed -i 's/+cu[0-9]\+//g' requirements.txt
