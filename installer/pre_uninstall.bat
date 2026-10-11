@echo off
:: ETSAM installer: pre-uninstall step (Windows).
:: Removes the folder with the etsam commands (added by post_install.bat) from
:: the user's PATH. PowerShell is called by its full path: the uninstaller runs
:: this script with a PATH that does not include PowerShell's folder.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command "$d = $env:PREFIX + '\commands'; $p = [Environment]::GetEnvironmentVariable('Path', 'User'); if ($p) { [Environment]::SetEnvironmentVariable('Path', ((($p -split ';') | Where-Object { $_ -and $_ -ne $d }) -join ';'), 'User') }"
exit /b 0
