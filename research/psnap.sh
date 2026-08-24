#!/bin/bash
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'cov_sat.py|f3_one.py|record_multiplicity' } | ForEach-Object { (\$_.CommandLine -split 'research/')[-1] }" 2>/dev/null | tr -d '\r' | grep -E '^(cov_sat|f3_one|record)' | sort | uniq -c
