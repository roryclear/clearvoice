# ClearVoice

<a href="https://apps.apple.com/gb/app/clearcam/id6743237694">
  <img src="https://developer.apple.com/assets/elements/badges/download-on-the-app-store.svg"
       alt="Download on the App Store"
       height="50"/>
</a>

## Python setup:
```
pip install -r requirements.txt
```
## Start voice cloning
```
python clearvoice.py
```
open localhost:8080

### for faster inference use tinygrad's BEAM search:
```
BEAM=2 python clearvoice.py
```
this will result in a longer initial run time as the searches are performed and cached. For visibility on the process use:
```
BEAM=2 DEBUG=2 python clearvoice.py
```
