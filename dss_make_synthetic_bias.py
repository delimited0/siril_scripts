import numpy as np
from astropy.io import fits

# --- Set these to match your camera ---
width = 6252      
height = 4176     
# pedestal ADU assumes gain of 126 in Player One Poseidon C Pro (IMX571) camera
# Computed from 8 * 126 + 0.8
pedestal_adu = 1009 

# Create constant 16-bit single-channel Bayer image (matches raw CFA format)
img = np.full((height, width), pedestal_adu, dtype=np.uint16)

# Write FITS
hdu = fits.PrimaryHDU(img)

# Optional but useful FITS header fields
hdu.header['BUNIT'] = 'ADU'
hdu.header['IMAGETYP'] = 'BIAS'
hdu.header['BAYERPAT'] = 'RGGB'   # IMX571 Bayer pattern — makes DSS show CFA=Yes
hdu.header['XBAYROFF'] = 0
hdu.header['YBAYROFF'] = 0
hdu.header['COMMENT'] = 'Synthetic constant bias frame for DSS'

hdu.writeto('synthetic_bias_master.fit', overwrite=True)
print("Wrote synthetic_bias_master.fit")