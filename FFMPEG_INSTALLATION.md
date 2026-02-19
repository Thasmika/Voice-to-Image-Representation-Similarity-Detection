# FFmpeg Installation Required

## Issue
The property tests are skipping because Whisper requires `ffmpeg` to process audio files, but it's not installed on your system.

## Error
```
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

This occurs when Whisper tries to use `ffmpeg` to load audio files.

## Solution: Install FFmpeg on Windows

### Option 1: Using Chocolatey (Recommended)
If you have Chocolatey installed:
```powershell
choco install ffmpeg
```

### Option 2: Using Scoop
If you have Scoop installed:
```powershell
scoop install ffmpeg
```

### Option 3: Manual Installation
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Download the "ffmpeg-release-essentials.zip" file
3. Extract the ZIP file to a location like `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your system PATH:
   - Open "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "System variables", find and select "Path"
   - Click "Edit"
   - Click "New"
   - Add `C:\ffmpeg\bin`
   - Click "OK" on all dialogs
5. **Restart your terminal/PowerShell** for the changes to take effect

### Verify Installation
After installing, verify ffmpeg is available:
```powershell
ffmpeg -version
```

You should see version information if it's installed correctly.

## After Installation
Once ffmpeg is installed, the property tests should run successfully:
```powershell
python -m pytest tests/test_api_properties.py::test_property_13_temporary_file_cleanup -v
```

The tests will no longer skip and will validate the resource management functionality properly.
