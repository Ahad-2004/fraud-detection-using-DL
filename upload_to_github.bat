@echo off

REM Check if Git is installed
git --version 2>nul
if errorlevel 1 (
    echo Git is not installed. Please download and install Git from:
    echo https://git-scm.com/downloads
    pause
    exit /b
)

REM Navigate to project directory
cd /d "%~dp0"

REM Initialize Git repository
echo Initializing Git repository...
git init

REM Add all files
echo Adding files to Git...
git add .

REM Commit changes
echo Committing changes...
git commit -m "Initial commit - Professional Insurance Fraud Detection System"

REM Rename branch to main
git branch -M main

REM Add remote repository
echo Adding GitHub repository...
git remote add origin https://github.com/Ahad-2004/fraud-detection-using-DL.git

REM Push changes
echo Pushing code to GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo Push failed. You may need to:
    echo 1. Create a personal access token (Classic) on GitHub with repo permissions
    echo 2. Use the token as your password when prompted
    echo.
    echo Create token here: https://github.com/settings/tokens/new
    pause
)

echo.
echo Success! Your code has been pushed to GitHub.
echo Repository: https://github.com/Ahad-2004/fraud-detection-using-DL
pause
