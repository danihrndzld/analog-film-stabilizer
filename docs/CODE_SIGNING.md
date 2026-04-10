# macOS Code Signing & Notarization

This document explains how to configure code signing so DMG downloads work without users needing to run `xattr -cr`.

## Why This Is Needed

macOS applies a quarantine attribute to downloaded files. Without proper code signing and notarization, Gatekeeper shows warnings like "app is damaged" or "cannot verify developer", forcing users to run `xattr -cr` to bypass the check.

**Solution:** Developer ID signing + Apple notarization + stapling.

## Requirements

- **Apple Developer Program membership** ($99/year) — required for Developer ID certificates and notarization
- A Mac with Xcode to generate the certificate (one-time setup)

## One-Time Setup (on a Mac)

### 1. Generate Developer ID Application Certificate

1. Open Keychain Access → Certificate Assistant → Request a Certificate from a Certificate Authority
2. Save the `.certSigningRequest` file
3. Go to [developer.apple.com/account/resources/certificates/add](https://developer.apple.com/account/resources/certificates/add)
4. Choose **Developer ID Application**, upload your CSR, download the `.cer`
5. Double-click the `.cer` to import into your login keychain

### 2. Export as P12

1. In Keychain Access → My Certificates
2. Find "Developer ID Application: Your Name"
3. Select the certificate AND its private key
4. Right-click → Export → save as `.p12` with a strong password

### 3. Base64-Encode the P12

```bash
openssl base64 -in DeveloperID.p12 -out DeveloperID.txt
# Copy the contents into a GitHub secret
```

### 4. Create App-Specific Password

1. Go to [appleid.apple.com](https://appleid.apple.com) → Security → App-Specific Passwords
2. Generate a new password for "GitHub CI"
3. Save it securely

### 5. Find Your Team ID

Log in to [developer.apple.com/account](https://developer.apple.com/account) — the Team ID is shown in the top-right membership details (10-character string like `AB12CD34EF`).

## GitHub Repository Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret:

| Secret Name | Value |
|-------------|-------|
| `CSC_CONTENT` | Contents of `DeveloperID.txt` (base64-encoded P12) |
| `CSC_KEY_PASSWORD` | Password you set when exporting the P12 |
| `APPLE_ID` | Your Apple Developer account email |
| `APPLE_APP_SPECIFIC_PASSWORD` | The app-specific password from step 4 |
| `APPLE_TEAM_ID` | Your 10-character Team ID |

## How It Works

When these secrets are configured:

1. CI imports the Developer ID certificate into a temporary keychain
2. electron-builder signs the app with hardened runtime enabled
3. `scripts/notarize.js` submits to Apple's notarization service (typically 1-5 min)
4. electron-builder staples the notarization ticket to the DMG
5. Users can download and run without any `xattr` commands

When secrets are NOT configured:

- CI falls back to ad-hoc signing (`--sign -`)
- A warning is logged
- Users will need to run `xattr -cr` on the downloaded app

## Verification

After a signed build, verify locally:

```bash
# Check code signature
codesign -dv --verbose=4 "dist/mac-arm64/Perforation Stabilizer.app"

# Check Gatekeeper assessment
spctl -a -vvv "dist/mac-arm64/Perforation Stabilizer.app"

# Check notarization staple
stapler validate "dist/Perforation Stabilizer-arm64.dmg"
```

## Troubleshooting

**Notarization fails with "invalid signature":**
- Ensure `hardenedRuntime: true` is set in `electron/package.json`
- Check that entitlements.mac.plist exists and is valid

**"The signature of the binary is invalid":**
- The Python binaries (`stabilizer_arm64`, `stabilizer_x64`) are unsigned but work because they're bundled as resources, not executables in the app bundle

**Notarization times out:**
- Apple's service occasionally has delays; the `@electron/notarize` package retries automatically
- Check [Apple System Status](https://developer.apple.com/system-status/) for notarization service outages

## References

- [Electron Code Signing Tutorial](https://www.electronjs.org/docs/latest/tutorial/code-signing)
- [@electron/notarize](https://github.com/electron/notarize)
- [electron-builder Code Signing (Mac)](https://www.electron.build/code-signing-mac.html)
